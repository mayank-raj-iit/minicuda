// main.cpp - the minicuda command-line driver.
//
//   minicuda kernel.cu -o kernel.ptx           compile to PTX (default -O2)
//   minicuda kernel.cu --dump-ir -O0           print the IR before optimization
//   minicuda kernel.cu --stats                 show what each optimization pass did
//   minicuda kernel.cu --run saxpy --grid 4 --block 256 --args in.txt --out out.txt
//                                              run a kernel on the CPU interpreter
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "diag.hpp"
#include "interp.hpp"
#include "irgen.hpp"
#include "lexer.hpp"
#include "opt.hpp"
#include "parser.hpp"
#include "ptxgen.hpp"
#include "sema.hpp"

namespace {

void usage() {
  std::cerr <<
      "usage: minicuda <input.cu> [options]\n"
      "  -o <file>          write PTX to <file> (default: stdout)\n"
      "  -O0 | -O1 | -O2    optimization level (default -O2)\n"
      "  --arch=sm_XX       PTX target (default sm_75, the Colab T4 GPU)\n"
      "  --lineinfo         emit .file/.loc source line information\n"
      "  --no-fmad          do not contract a*b+c into fma\n"
      "  --dump-tokens | --dump-ast | --dump-ir   print an intermediate stage and stop\n"
      "  --stats            print IR size after each optimization pass\n"
      "  --run <kernel>     run on the CPU interpreter instead of emitting PTX; needs:\n"
      "      --grid X[,Y[,Z]] --block X[,Y[,Z]] --args <file> [--out <file>]\n";
}

std::string readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open '" + path + "'");
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

Dim3 parseDim(const std::string& s) {
  Dim3 d;
  unsigned v[3] = {1, 1, 1};
  int n = std::sscanf(s.c_str(), "%u,%u,%u", &v[0], &v[1], &v[2]);
  if (n < 1 || v[0] == 0 || v[1] == 0 || v[2] == 0) throw std::runtime_error("bad dimension '" + s + "'");
  d.x = v[0]; d.y = v[1]; d.z = v[2];
  return d;
}

// Argument file format (whitespace separated, '#' starts a comment):
//   i32 256                 an int argument
//   f32 2.5                 a float argument
//   buf f32 4 1 2 3 4       a device buffer of 4 floats with these values
//   buf i32 3 7 8 9         a device buffer of 3 ints
//   zeros f32 256           a device buffer of 256 zeros
struct HostBuffer { int param; IRType type; size_t count; uint64_t addr; };

std::vector<KernelArg> loadArgs(const std::string& path, DeviceMemory& mem, std::vector<HostBuffer>& bufs) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open args file '" + path + "'");
  std::stringstream clean;
  for (std::string line; std::getline(in, line);) {
    auto hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    clean << line << "\n";
  }
  std::vector<KernelArg> args;
  std::string kind;
  while (clean >> kind) {
    KernelArg a;
    if (kind == "i32") { a.type = IRType::I32; clean >> a.i; }
    else if (kind == "f32") { a.type = IRType::F32; clean >> a.f; }
    else if (kind == "buf" || kind == "zeros") {
      std::string ty;
      size_t n;
      clean >> ty >> n;
      if (ty != "f32" && ty != "i32") throw std::runtime_error("buffer type must be f32 or i32");
      a.type = IRType::I64;
      a.ptr = mem.alloc(n * 4);
      uint8_t* p = mem.find(a.ptr, n * 4)->bytes.data();
      if (kind == "buf")
        for (size_t i = 0; i < n; i++) {
          if (ty == "f32") { float v; clean >> v; std::memcpy(p + 4 * i, &v, 4); }
          else { int32_t v; clean >> v; std::memcpy(p + 4 * i, &v, 4); }
        }
      bufs.push_back({(int)args.size(), ty == "f32" ? IRType::F32 : IRType::I32, n, a.ptr});
    } else {
      throw std::runtime_error("unknown argument kind '" + kind + "' in args file");
    }
    if (!clean) throw std::runtime_error("malformed args file '" + path + "'");
    args.push_back(a);
  }
  return args;
}

void writeBuffers(std::ostream& os, DeviceMemory& mem, const std::vector<HostBuffer>& bufs) {
  for (const auto& b : bufs) {
    os << "buf " << b.param << " " << (b.type == IRType::F32 ? "f32" : "i32") << " " << b.count << "\n";
    const uint8_t* p = mem.find(b.addr, b.count * 4)->bytes.data();
    for (size_t i = 0; i < b.count; i++) {
      char tmp[32];
      if (b.type == IRType::F32) { float v; std::memcpy(&v, p + 4 * i, 4); std::snprintf(tmp, sizeof tmp, "%.9g", v); }
      else { int32_t v; std::memcpy(&v, p + 4 * i, 4); std::snprintf(tmp, sizeof tmp, "%d", v); }
      os << tmp << ((i + 1) % 8 == 0 || i + 1 == b.count ? "\n" : " ");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string input, output, runKernel, argsFile, outFile, arch = "sm_75";
  std::string gridStr = "1", blockStr = "1";
  int level = 2;
  bool lineInfo = false, fmad = true, dumpTokens = false, dumpAstFlag = false, dumpIR = false, stats = false;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << "error: " << a << " needs a value\n"; std::exit(2); }
      return argv[++i];
    };
    if (a == "-o") output = next();
    else if (a == "-O0" || a == "-O1" || a == "-O2" || a == "-O3") level = std::min(2, a[2] - '0');
    else if (a.rfind("--arch=", 0) == 0) arch = a.substr(7);
    else if (a == "--lineinfo") lineInfo = true;
    else if (a == "--no-fmad") fmad = false;
    else if (a == "--dump-tokens") dumpTokens = true;
    else if (a == "--dump-ast") dumpAstFlag = true;
    else if (a == "--dump-ir") dumpIR = true;
    else if (a == "--stats") stats = true;
    else if (a == "--run") runKernel = next();
    else if (a == "--grid") gridStr = next();
    else if (a == "--block") blockStr = next();
    else if (a == "--args") argsFile = next();
    else if (a == "--out") outFile = next();
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else if (!a.empty() && a[0] == '-') { std::cerr << "error: unknown option '" << a << "'\n"; usage(); return 2; }
    else input = a;
  }
  if (input.empty()) { usage(); return 2; }

  std::string source;
  try {
    source = readFile(input);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  try {
    // Phase 1: lexing
    std::vector<Token> tokens = tokenize(source);
    if (dumpTokens) {
      for (const Token& t : tokens) std::cout << t.line << ":" << t.col << "\t" << t.text << "\n";
      return 0;
    }
    // Phase 2: parsing
    Program prog = parseProgram(tokens);
    // Phase 3: semantic analysis
    checkProgram(prog);
    if (dumpAstFlag) { dumpAst(prog, std::cout); return 0; }
    if (prog.kernels.empty()) throw CompileError(1, 1, "no __global__ kernels found");

    // Phase 4: IR generation, Phase 5: optimization
    std::vector<Function> fns;
    for (auto& k : prog.kernels) {
      Function f = generateIR(*k);
      OptOptions oo;
      oo.level = level;
      oo.fmad = fmad && level > 0;
      auto st = optimize(f, oo);
      if (stats) {
        std::cerr << "kernel " << f.name << ":\n";
        for (auto& s : st) std::cerr << "  " << s.pass << ": " << s.instrs << " IR instructions\n";
      }
      fns.push_back(std::move(f));
    }
    if (dumpIR) {
      for (auto& f : fns) printFunction(f, std::cout);
      return 0;
    }

    if (!runKernel.empty()) {  // execute on the CPU interpreter
      const Function* fn = nullptr;
      for (auto& f : fns) if (f.name == runKernel) fn = &f;
      if (!fn) throw std::runtime_error("no kernel named '" + runKernel + "'");
      if (argsFile.empty()) throw std::runtime_error("--run needs --args <file>");
      DeviceMemory mem;
      std::vector<HostBuffer> bufs;
      std::vector<KernelArg> args = loadArgs(argsFile, mem, bufs);
      for (size_t i = 0; i < args.size() && i < fn->params.size(); i++) {
        IRType want = fn->params[i].type;
        if (args[i].type != want)
          throw std::runtime_error("argument " + std::to_string(i) + " ('" + fn->params[i].name + "') should be " +
                                   (fn->params[i].isPtr ? "a buffer" : typeName(want)));
      }
      RunStats rs = interpret(*fn, mem, parseDim(gridStr), parseDim(blockStr), args);
      if (stats) std::cerr << "executed " << rs.instructions << " IR instructions, " << rs.barriers << " barriers\n";
      if (outFile.empty()) writeBuffers(std::cout, mem, bufs);
      else { std::ofstream os(outFile); writeBuffers(os, mem, bufs); }
      return 0;
    }

    // Phase 6: PTX emission
    PtxOptions po;
    po.arch = arch;
    po.lineInfo = lineInfo;
    po.sourceName = input;
    size_t count = 0;
    std::string ptx = emitPtx(fns, po, &count);
    if (stats) std::cerr << "PTX instructions: " << count << "\n";
    if (output.empty()) std::cout << ptx;
    else {
      std::ofstream os(output);
      if (!os) throw std::runtime_error("cannot write '" + output + "'");
      os << ptx;
    }
    return 0;
  } catch (const CompileError& e) {
    std::cerr << input << ":" << e.line << ":" << e.col << ": error: " << e.what() << "\n";
    return 1;
  } catch (const RunError& e) {
    std::cerr << "runtime error: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
