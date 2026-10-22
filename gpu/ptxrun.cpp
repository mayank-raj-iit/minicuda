// ptxrun.cpp - load a .ptx file with the CUDA Driver API and run one kernel.
//
// This is the same path a real compiler toolchain uses at run time: the
// driver JIT-compiles PTX into the GPU's machine code (SASS), then launches it.
//
// Build (Google Colab or any Linux box with the CUDA toolkit):
//   nvcc -O2 -o gpu/ptxrun gpu/ptxrun.cpp -lcuda
// Run:
//   gpu/ptxrun kernel.ptx saxpy 4,1,1 256,1,1 args.txt out.txt
// The args/out files use the same format as `minicuda --run`.
#include <cuda.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(call)                                                              \
  do {                                                                           \
    CUresult r_ = (call);                                                        \
    if (r_ != CUDA_SUCCESS) {                                                    \
      const char* name_ = nullptr;                                               \
      cuGetErrorName(r_, &name_);                                                \
      std::fprintf(stderr, "%s failed: %s\n", #call, name_ ? name_ : "unknown"); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

struct Arg {
  enum Kind { I32, F32, Buffer } kind;
  int32_t i = 0;
  float f = 0;
  bool isFloat = false;          // buffer element type
  std::vector<uint32_t> host;    // buffer contents (raw 32-bit words)
  CUdeviceptr dev = 0;
};

static void parseDims(const char* s, unsigned d[3]) {
  d[0] = d[1] = d[2] = 1;
  if (std::sscanf(s, "%u,%u,%u", &d[0], &d[1], &d[2]) < 1) {
    std::fprintf(stderr, "bad dimensions '%s'\n", s);
    std::exit(2);
  }
}

static std::vector<Arg> loadArgs(const char* path) {
  std::ifstream in(path);
  if (!in) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
  std::stringstream clean;
  for (std::string line; std::getline(in, line);) {
    auto h = line.find('#');
    if (h != std::string::npos) line.resize(h);
    clean << line << "\n";
  }
  std::vector<Arg> args;
  std::string kind;
  while (clean >> kind) {
    Arg a;
    if (kind == "i32") { a.kind = Arg::I32; clean >> a.i; }
    else if (kind == "f32") { a.kind = Arg::F32; clean >> a.f; }
    else if (kind == "buf" || kind == "zeros") {
      std::string ty;
      size_t n;
      clean >> ty >> n;
      a.kind = Arg::Buffer;
      a.isFloat = ty == "f32";
      a.host.assign(n, 0);
      if (kind == "buf")
        for (size_t k = 0; k < n; k++) {
          if (a.isFloat) { float v; clean >> v; std::memcpy(&a.host[k], &v, 4); }
          else { int32_t v; clean >> v; std::memcpy(&a.host[k], &v, 4); }
        }
    } else {
      std::fprintf(stderr, "unknown argument kind '%s'\n", kind.c_str());
      std::exit(2);
    }
    args.push_back(std::move(a));
  }
  return args;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: ptxrun <file.ptx> <kernel> <gx,gy,gz> <bx,by,bz> <args.txt> <out.txt>\n");
    return 2;
  }
  std::ifstream pf(argv[1]);
  if (!pf) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
  std::stringstream ss;
  ss << pf.rdbuf();
  std::string ptx = ss.str();
  unsigned grid[3], block[3];
  parseDims(argv[3], grid);
  parseDims(argv[4], block);
  std::vector<Arg> args = loadArgs(argv[5]);

  // 1. pick GPU 0 and make its primary context current
  CHECK(cuInit(0));
  CUdevice dev;
  CHECK(cuDeviceGet(&dev, 0));
  char name[256];
  int major = 0, minor = 0;
  CHECK(cuDeviceGetName(name, sizeof name, dev));
  CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CHECK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  CUcontext ctx;
  CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK(cuCtxSetCurrent(ctx));

  // 2. JIT-compile the PTX. On failure the driver's ptxas error log tells us why.
  char errLog[16384] = {0}, infoLog[16384] = {0};
  CUjit_option opts[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                         CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE};
  void* optVals[] = {errLog, (void*)(uintptr_t)sizeof errLog, infoLog, (void*)(uintptr_t)sizeof infoLog,
                     (void*)(uintptr_t)1};
  CUmodule mod;
  CUresult jit = cuModuleLoadDataEx(&mod, ptx.c_str(), 5, opts, optVals);
  if (jit != CUDA_SUCCESS) {
    std::fprintf(stderr, "PTX JIT compilation failed on %s (sm_%d%d):\n%s\n", name, major, minor, errLog);
    return 1;
  }
  if (std::getenv("PTXRUN_VERBOSE")) std::fprintf(stderr, "JIT log (%s, sm_%d%d):\n%s\n", name, major, minor, infoLog);
  CUfunction fn;
  CHECK(cuModuleGetFunction(&fn, mod, argv[2]));

  // 3. copy buffers to the GPU and build the kernel parameter array
  std::vector<void*> params(args.size());
  for (size_t k = 0; k < args.size(); k++) {
    Arg& a = args[k];
    if (a.kind == Arg::Buffer) {
      size_t bytes = a.host.size() * 4;
      CHECK(cuMemAlloc(&a.dev, bytes ? bytes : 4));
      if (bytes) CHECK(cuMemcpyHtoD(a.dev, a.host.data(), bytes));
      params[k] = &a.dev;
    } else if (a.kind == Arg::I32) {
      params[k] = &a.i;
    } else {
      params[k] = &a.f;
    }
  }

  // 4. launch and time it
  CUevent start, stop;
  CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
  CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
  CHECK(cuEventRecord(start, 0));
  CHECK(cuLaunchKernel(fn, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, 0, params.data(), nullptr));
  CHECK(cuEventRecord(stop, 0));
  CHECK(cuCtxSynchronize());
  float ms = 0;
  CHECK(cuEventElapsedTime(&ms, start, stop));
  std::fprintf(stderr, "%s on %s (sm_%d%d): %.3f ms\n", argv[2], name, major, minor, ms);

  // 5. copy results back and write them in the shared text format
  FILE* out = std::fopen(argv[6], "w");
  if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[6]); return 2; }
  for (size_t k = 0; k < args.size(); k++) {
    Arg& a = args[k];
    if (a.kind != Arg::Buffer) continue;
    size_t n = a.host.size();
    if (n) CHECK(cuMemcpyDtoH(a.host.data(), a.dev, n * 4));
    std::fprintf(out, "buf %zu %s %zu\n", k, a.isFloat ? "f32" : "i32", n);
    for (size_t e = 0; e < n; e++) {
      if (a.isFloat) { float v; std::memcpy(&v, &a.host[e], 4); std::fprintf(out, "%.9g", v); }
      else { int32_t v; std::memcpy(&v, &a.host[e], 4); std::fprintf(out, "%d", v); }
      std::fputc((e + 1) % 8 == 0 || e + 1 == n ? '\n' : ' ', out);
    }
    CHECK(cuMemFree(a.dev));
  }
  std::fclose(out);
  CHECK(cuModuleUnload(mod));
  CHECK(cuDevicePrimaryCtxRelease(dev));
  return 0;
}
