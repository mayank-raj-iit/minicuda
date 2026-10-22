// ptxgen.cpp - turn IR into PTX text.
//
// PTX is NVIDIA's virtual instruction set. It has typed instructions
// (add.s32, mul.rn.f32, ld.global.f32 ...) and an unlimited supply of
// virtual registers, declared up front:  .reg .b32 %r<12>;
// ptxas (or the driver's JIT) later turns PTX into real GPU machine code (SASS).
#include "ptxgen.hpp"

#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

namespace {

Instr movInstr(int dst, Operand src, const Instr& at) {
  Instr m;
  m.op = Opcode::Mov;
  m.type = src.type;
  m.dst = dst;
  m.src = {src};
  m.line = at.line;
  m.col = at.col;
  return m;
}

Opcode swapCompare(Opcode op) {  // a < b  <=>  b > a
  switch (op) {
    case Opcode::CmpLT: return Opcode::CmpGT;
    case Opcode::CmpLE: return Opcode::CmpGE;
    case Opcode::CmpGT: return Opcode::CmpLT;
    case Opcode::CmpGE: return Opcode::CmpLE;
    default: return op;
  }
}

bool isBinaryOp(Opcode op) {
  switch (op) {
    case Opcode::Add: case Opcode::Sub: case Opcode::Mul: case Opcode::Div: case Opcode::Rem:
    case Opcode::And: case Opcode::Or: case Opcode::Xor: case Opcode::Shl: case Opcode::Shr:
    case Opcode::Min: case Opcode::Max: case Opcode::MulWide:
      return true;
    default:
      return isCompare(op);
  }
}

}  // namespace

void legalizeForPtx(Function& f) {
  for (auto& blk : f.blocks) {
    std::vector<Instr> out;
    for (Instr in : blk.instrs) {
      auto materialize = [&](Operand& o) {
        if (!o.isImm()) return;
        IRType t = o.type;
        if (t == IRType::Pred) {  // PTX has no predicate immediates: build one with setp
          int r = f.newReg(IRType::I32);
          out.push_back(movInstr(r, Operand::I(o.imm ? 1 : 0), in));
          int p = f.newReg(IRType::Pred);
          Instr c;
          c.op = Opcode::CmpNE; c.type = IRType::I32; c.dst = p;
          c.src = {Operand::R(r, IRType::I32), Operand::I(0)};
          c.line = in.line; c.col = in.col;
          out.push_back(c);
          o = Operand::R(p, IRType::Pred);
          return;
        }
        int r = f.newReg(t);
        out.push_back(movInstr(r, o, in));
        o = Operand::R(r, t);
      };
      auto& s = in.src;
      if (in.op == Opcode::CondBr && s[0].isImm()) {  // constant branch (only at -O0)
        in.op = Opcode::Br;
        if (!s[0].imm) in.target = in.target2;
        in.target2 = -1;
        s.clear();
      } else if (isBinaryOp(in.op)) {
        if (s[0].isImm()) {
          if (isCommutative(in.op)) std::swap(s[0], s[1]);
          else if (isCompare(in.op)) { std::swap(s[0], s[1]); in.op = swapCompare(in.op); }
        }
        materialize(s[0]);
        if (in.type == IRType::Pred) materialize(s[1]);
      } else {
        switch (in.op) {
          case Opcode::Neg: case Opcode::Abs: case Opcode::Sqrt: case Opcode::Not:
          case Opcode::CvtI2F: case Opcode::CvtF2I: case Opcode::CvtI2L: case Opcode::CvtaGlobal:
            materialize(s[0]);
            break;
          case Opcode::Fma:
            for (auto& o : s) materialize(o);
            break;
          case Opcode::Select:
            materialize(s[0]);
            break;
          case Opcode::LdGlobal: case Opcode::LdShared:
            materialize(s[0]);
            break;
          case Opcode::StGlobal: case Opcode::StShared:
          case Opcode::AtomAddGlobal: case Opcode::AtomAddShared:
            materialize(s[0]);
            materialize(s[1]);
            break;
          case Opcode::Mov:
            if (in.type == IRType::Pred) materialize(s[0]);
            break;
          default:
            break;
        }
      }
      out.push_back(std::move(in));
    }
    blk.instrs = std::move(out);
  }
}

std::string ptxVersionFor(const std::string& arch) {
  static const std::map<std::string, std::string> v = {
      {"sm_70", "7.0"}, {"sm_72", "7.0"}, {"sm_75", "7.0"}, {"sm_80", "7.0"}, {"sm_86", "7.1"},
      {"sm_87", "7.4"}, {"sm_89", "7.8"}, {"sm_90", "7.8"}, {"sm_100", "8.6"}, {"sm_120", "8.7"}};
  auto it = v.find(arch);
  return it == v.end() ? "7.0" : it->second;
}

namespace {

class Emitter {
 public:
  Emitter(Function& f, int index, const PtxOptions& o, std::ostream& os) : f_(f), idx_(index), opt_(o), os_(os) {}

  size_t run() {
    nameRegisters();
    std::vector<char> isTarget(f_.blocks.size(), 0);
    std::vector<char> used(f_.vregs.size(), 0);
    for (size_t b = 0; b < f_.blocks.size(); b++)
      for (auto& in : f_.blocks[b].instrs) {
        // a block needs a label only if some branch actually names it
        // (falling through to the next block needs no label)
        const int next = (int)b + 1;
        if (in.op == Opcode::Br && in.target != next) isTarget[in.target] = 1;
        if (in.op == Opcode::CondBr) {
          if (in.target2 == next) isTarget[in.target] = 1;
          else if (in.target == next) isTarget[in.target2] = 1;
          else { isTarget[in.target] = 1; isTarget[in.target2] = 1; }
        }
        for (auto& o : in.src) if (o.isReg()) used[o.reg] = 1;
      }

    os_ << "\t// .globl\t" << f_.name << "\n";
    os_ << ".visible .entry " << f_.name << "(";
    for (size_t i = 0; i < f_.params.size(); i++) {
      os_ << (i ? ",\n" : "\n") << "\t.param " << paramType(f_.params[i]) << " " << paramName((int)i);
    }
    os_ << (f_.params.empty() ? ")\n{\n" : "\n)\n{\n");
    static const char* cls[] = {"%p", "%r", "%f", "%rd"};
    static const char* decl[] = {".pred", ".b32", ".f32", ".b64"};
    for (int c = 0; c < 4; c++)
      if (count_[c] > 0) os_ << "\t.reg " << decl[c] << " \t" << cls[c] << "<" << count_[c] + 1 << ">;\n";
    for (const auto& s : f_.shared)
      os_ << "\t.shared .align 4 .b8 " << s.name << "[" << s.count * typeSize(s.elem) << "];\n";
    os_ << "\n";

    size_t n = 0;
    int lastLine = -1;
    for (size_t b = 0; b < f_.blocks.size(); b++) {
      if (isTarget[b]) os_ << label((int)b) << ":\n";
      for (const Instr& in : f_.blocks[b].instrs) {
        if (opt_.lineInfo && in.line > 0 && in.line != lastLine) {
          os_ << "\t.loc\t1 " << in.line << " " << in.col << "\n";
          lastLine = in.line;
        }
        n += instr(in, (int)b, used);
      }
    }
    os_ << "\n}\n";
    return n;
  }

 private:
  Function& f_;
  int idx_;
  const PtxOptions& opt_;
  std::ostream& os_;
  std::vector<std::string> regName_;
  int count_[4] = {0, 0, 0, 0};  // pred, b32, f32, b64

  static int classOf(IRType t) {
    switch (t) {
      case IRType::Pred: return 0;
      case IRType::I32: return 1;
      case IRType::F32: return 2;
      default: return 3;
    }
  }
  void nameRegisters() {
    static const char* cls[] = {"%p", "%r", "%f", "%rd"};
    regName_.assign(f_.vregs.size(), "");
    auto name = [&](int r) {
      if (!regName_[r].empty()) return;
      int c = classOf(f_.vregs[r].type);
      regName_[r] = std::string(cls[c]) + std::to_string(++count_[c]);
    };
    for (auto& blk : f_.blocks)
      for (auto& in : blk.instrs) {
        for (auto& o : in.src) if (o.isReg()) name(o.reg);
        if (in.dst >= 0) name(in.dst);
      }
  }

  std::string paramName(int i) const { return f_.name + "_param_" + std::to_string(i); }
  static const char* paramType(const ParamInfo& p) {
    return p.type == IRType::F32 ? ".f32" : p.type == IRType::I64 ? ".u64" : ".u32";
  }
  std::string label(int b) const { return "$L__BB" + std::to_string(idx_) + "_" + std::to_string(b); }

  std::string opnd(const Operand& o) const {
    if (o.isReg()) return regName_[o.reg];
    if (o.kind == Operand::ImmFloat) {  // exact bit pattern: 0f3F800000 == 1.0f
      uint32_t bits;
      std::memcpy(&bits, &o.fimm, 4);
      char buf[16];
      std::snprintf(buf, sizeof buf, "0f%08X", bits);
      return buf;
    }
    return std::to_string(o.imm);
  }
  std::string dst(const Instr& in) const { return regName_[in.dst]; }

  // integer ops use .s32 for arithmetic and .b32 for bitwise; float ops use .f32
  static std::string sfx(IRType t, bool bitwise = false) {
    switch (t) {
      case IRType::F32: return ".f32";
      case IRType::I64: return bitwise ? ".b64" : ".s64";
      case IRType::Pred: return ".pred";
      default: return bitwise ? ".b32" : ".s32";
    }
  }

  void line(const std::string& op, const std::vector<std::string>& args) {
    os_ << "\t" << op << " \t";
    for (size_t i = 0; i < args.size(); i++) os_ << (i ? ", " : "") << args[i];
    os_ << ";\n";
  }

  size_t instr(const Instr& in, int b, const std::vector<char>& used) {
    const auto& s = in.src;
    const IRType t = in.type;
    const bool f32 = t == IRType::F32;
    switch (in.op) {
      case Opcode::Mov:
        line(std::string("mov") + (t == IRType::F32 ? ".f32" : t == IRType::I64 ? ".u64" : t == IRType::Pred ? ".pred" : ".u32"),
             {dst(in), opnd(s[0])});
        return 1;
      case Opcode::Add: line(f32 ? "add.rn.f32" : "add" + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Sub: line(f32 ? "sub.rn.f32" : "sub" + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Mul: line(f32 ? "mul.rn.f32" : "mul.lo" + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Div: line(f32 ? "div.rn.f32" : "div" + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Rem: line("rem" + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::And: line("and" + sfx(t, true), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Or: line("or" + sfx(t, true), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Xor: line("xor" + sfx(t, true), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Shl: line("shl.b32", {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Shr: line("shr.s32", {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Min: line("min" + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Max: line("max" + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::Neg: line("neg" + sfx(t), {dst(in), opnd(s[0])}); return 1;
      case Opcode::Abs: line("abs" + sfx(t), {dst(in), opnd(s[0])}); return 1;
      case Opcode::Not: line("not.pred", {dst(in), opnd(s[0])}); return 1;
      case Opcode::Sqrt: line("sqrt.rn.f32", {dst(in), opnd(s[0])}); return 1;
      case Opcode::Fma: line(f32 ? "fma.rn.f32" : "mad.lo.s32", {dst(in), opnd(s[0]), opnd(s[1]), opnd(s[2])}); return 1;
      case Opcode::CmpLT: case Opcode::CmpLE: case Opcode::CmpGT:
      case Opcode::CmpGE: case Opcode::CmpEQ: case Opcode::CmpNE: {
        static const char* c[] = {"lt", "le", "gt", "ge", "eq", "ne"};
        std::string cmp = c[(int)in.op - (int)Opcode::CmpLT];
        if (f32 && in.op == Opcode::CmpNE) cmp = "neu";  // C's != is true for NaN
        line("setp." + cmp + sfx(t), {dst(in), opnd(s[0]), opnd(s[1])});
        return 1;
      }
      case Opcode::Select:
        line(std::string("selp") + (f32 ? ".f32" : t == IRType::I64 ? ".b64" : ".b32"),
             {dst(in), opnd(s[1]), opnd(s[2]), opnd(s[0])});
        return 1;
      case Opcode::CvtI2F: line("cvt.rn.f32.s32", {dst(in), opnd(s[0])}); return 1;
      case Opcode::CvtF2I: line("cvt.rzi.s32.f32", {dst(in), opnd(s[0])}); return 1;
      case Opcode::CvtI2L: line("cvt.s64.s32", {dst(in), opnd(s[0])}); return 1;
      case Opcode::MulWide: line("mul.wide.s32", {dst(in), opnd(s[0]), opnd(s[1])}); return 1;
      case Opcode::LdParam:
        line(std::string("ld.param") + paramType(f_.params[in.aux]), {dst(in), "[" + paramName(in.aux) + "]"});
        return 1;
      case Opcode::CvtaGlobal: line("cvta.to.global.u64", {dst(in), opnd(s[0])}); return 1;
      case Opcode::SReg: line("mov.u32", {dst(in), sregName(in.aux)}); return 1;
      case Opcode::SharedAddr: line("mov.u32", {dst(in), f_.shared[in.aux].name}); return 1;
      case Opcode::LdGlobal: line(f32 ? "ld.global.f32" : "ld.global.u32", {dst(in), "[" + opnd(s[0]) + "]"}); return 1;
      case Opcode::LdShared: line(f32 ? "ld.shared.f32" : "ld.shared.u32", {dst(in), "[" + opnd(s[0]) + "]"}); return 1;
      case Opcode::StGlobal: line(f32 ? "st.global.f32" : "st.global.u32", {"[" + opnd(s[0]) + "]", opnd(s[1])}); return 1;
      case Opcode::StShared: line(f32 ? "st.shared.f32" : "st.shared.u32", {"[" + opnd(s[0]) + "]", opnd(s[1])}); return 1;
      case Opcode::AtomAddGlobal: case Opcode::AtomAddShared: {
        std::string space = in.op == Opcode::AtomAddGlobal ? ".global" : ".shared";
        std::string ty = f32 ? ".f32" : ".u32";
        if (!used[in.dst]) line("red" + space + ".add" + ty, {"[" + opnd(s[0]) + "]", opnd(s[1])});  // result unused
        else line("atom" + space + ".add" + ty, {dst(in), "[" + opnd(s[0]) + "]", opnd(s[1])});
        return 1;
      }
      case Opcode::Bar: line("bar.sync", {"0"}); return 1;
      case Opcode::Br:
        if (in.target == b + 1) return 0;  // falls through to the next block
        line("bra.uni", {label(in.target)});
        return 1;
      case Opcode::CondBr: {
        std::string p = opnd(s[0]);
        if (in.target2 == b + 1) { line("@" + p + " bra", {label(in.target)}); return 1; }
        if (in.target == b + 1) { line("@!" + p + " bra", {label(in.target2)}); return 1; }
        line("@" + p + " bra", {label(in.target)});
        line("bra.uni", {label(in.target2)});
        return 2;
      }
      case Opcode::Ret: os_ << "\tret;\n"; return 1;
    }
    return 0;
  }
};

}  // namespace

std::string emitPtx(std::vector<Function>& fns, const PtxOptions& opt, size_t* instrCount) {
  std::ostringstream os;
  os << "//\n// Generated by MiniCUDA - a teaching compiler for a CUDA C subset\n//\n\n";
  os << ".version " << ptxVersionFor(opt.arch) << "\n";
  os << ".target " << opt.arch << "\n";
  os << ".address_size 64\n\n";
  if (opt.lineInfo) os << "\t.file\t1 \"" << opt.sourceName << "\"\n\n";
  size_t total = 0;
  for (size_t i = 0; i < fns.size(); i++) {
    legalizeForPtx(fns[i]);
    if (i) os << "\n";
    total += Emitter(fns[i], (int)i, opt, os).run();
  }
  if (instrCount) *instrCount = total;
  return os.str();
}
