// interp.cpp - the IR interpreter ("a tiny GPU simulator").
//
// Threads of one block are run one after another. When a thread reaches
// __syncthreads() it pauses; once every live thread of the block is paused
// at the barrier, they are all released. This is enough to model shared
// memory + barriers correctly for race-free kernels.
#include "interp.hpp"

#include <climits>
#include <cmath>
#include <cstring>
#include <sstream>

uint64_t DeviceMemory::alloc(size_t bytes) {
  Buffer b;
  b.base = nextBase;
  b.bytes.assign(bytes, 0);
  nextBase += ((bytes + 4095) / 4096 + 1) * 4096;  // leave a gap so overruns are caught
  buffers.push_back(std::move(b));
  return buffers.back().base;
}

DeviceMemory::Buffer* DeviceMemory::find(uint64_t addr, size_t size) {
  for (auto& b : buffers)
    if (addr >= b.base && addr + size <= b.base + b.bytes.size()) return &b;
  return nullptr;
}

namespace {

// One register value. Only the field matching the register's type is used.
struct Val {
  int32_t i = 0;
  float f = 0;
  int64_t l = 0;
  bool p = false;
};

struct Thread {
  unsigned tid[3] = {0, 0, 0};
  std::vector<Val> regs;
  int bb = 0;
  size_t pc = 0;
  bool done = false, atBarrier = false;
};

int32_t wrapI32(int64_t v) { return (int32_t)(uint32_t)(uint64_t)v; }

class Interp {
 public:
  Interp(const Function& f, DeviceMemory& m, Dim3 g, Dim3 b, const std::vector<KernelArg>& a)
      : f_(f), mem_(m), grid_(g), block_(b), args_(a) {}

  RunStats run() {
    if (args_.size() != f_.params.size())
      throw RunError("kernel '" + f_.name + "' expects " + std::to_string(f_.params.size()) +
                     " arguments, got " + std::to_string(args_.size()));
    // shared memory layout: arrays one after another, 4-byte aligned
    size_t off = 0;
    for (const auto& s : f_.shared) {
      sharedOff_.push_back((uint32_t)off);
      off += ((size_t)s.count * typeSize(s.elem) + 3) & ~size_t(3);
    }
    sharedBytes_ = off;
    for (unsigned bz = 0; bz < grid_.z; bz++)
      for (unsigned by = 0; by < grid_.y; by++)
        for (unsigned bx = 0; bx < grid_.x; bx++) runBlock(bx, by, bz);
    return stats_;
  }

 private:
  const Function& f_;
  DeviceMemory& mem_;
  Dim3 grid_, block_;
  const std::vector<KernelArg>& args_;
  std::vector<uint32_t> sharedOff_;
  size_t sharedBytes_ = 0;
  std::vector<uint8_t> shared_;
  unsigned ctaid_[3] = {0, 0, 0};
  RunStats stats_;
  static constexpr uint64_t kStepLimit = 2000000000ULL;

  void runBlock(unsigned bx, unsigned by, unsigned bz) {
    ctaid_[0] = bx; ctaid_[1] = by; ctaid_[2] = bz;
    shared_.assign(sharedBytes_, 0);
    std::vector<Thread> threads;
    for (unsigned z = 0; z < block_.z; z++)
      for (unsigned y = 0; y < block_.y; y++)
        for (unsigned x = 0; x < block_.x; x++) {
          Thread t;
          t.tid[0] = x; t.tid[1] = y; t.tid[2] = z;
          t.regs.resize(f_.vregs.size());
          threads.push_back(std::move(t));
        }
    while (true) {
      for (auto& t : threads)
        if (!t.done && !t.atBarrier) runThread(t);
      bool waiting = false;
      for (auto& t : threads) waiting |= t.atBarrier;
      if (!waiting) return;               // every thread finished
      stats_.barriers++;
      for (auto& t : threads) t.atBarrier = false;  // release the barrier
    }
  }

  std::string where(const Thread& t) const {
    std::ostringstream os;
    os << "kernel '" << f_.name << "', block (" << ctaid_[0] << "," << ctaid_[1] << "," << ctaid_[2]
       << "), thread (" << t.tid[0] << "," << t.tid[1] << "," << t.tid[2] << ")";
    return os.str();
  }

  // ---- reading operands ----
  int32_t I(const Thread& t, const Operand& o) const { return o.isReg() ? t.regs[o.reg].i : (int32_t)o.imm; }
  float F(const Thread& t, const Operand& o) const { return o.isReg() ? t.regs[o.reg].f : o.fimm; }
  int64_t L(const Thread& t, const Operand& o) const { return o.isReg() ? t.regs[o.reg].l : o.imm; }
  bool P(const Thread& t, const Operand& o) const { return o.isReg() ? t.regs[o.reg].p : o.imm != 0; }

  uint8_t* globalPtr(const Thread& t, uint64_t addr, size_t n, const char* what) {
    DeviceMemory::Buffer* b = mem_.find(addr, n);
    if (!b) {
      std::ostringstream os;
      os << "out-of-bounds global " << what << " of " << n << " bytes at address 0x" << std::hex << addr
         << std::dec << " in " << where(t);
      throw RunError(os.str());
    }
    return b->bytes.data() + (addr - b->base);
  }
  uint8_t* sharedPtr(const Thread& t, int64_t addr, size_t n, const char* what) {
    if (addr < 0 || (size_t)addr + n > shared_.size()) {
      std::ostringstream os;
      os << "out-of-bounds shared " << what << " at offset " << addr << " (shared size " << shared_.size()
         << " bytes) in " << where(t);
      throw RunError(os.str());
    }
    return shared_.data() + addr;
  }

  void runThread(Thread& t) {
    while (true) {
      if (++stats_.instructions > kStepLimit) throw RunError("step limit exceeded (infinite loop?) in " + where(t));
      const Instr& in = f_.blocks[t.bb].instrs[t.pc];
      const auto& s = in.src;
      Val r;  // result
      bool write = in.dst >= 0;
      switch (in.op) {
        case Opcode::Mov:
          r = s[0].isReg() ? t.regs[s[0].reg] : Val{};
          if (!s[0].isReg()) { r.i = (int32_t)s[0].imm; r.l = s[0].imm; r.f = s[0].fimm; r.p = s[0].imm != 0; }
          break;
        case Opcode::Add: case Opcode::Sub: case Opcode::Mul: case Opcode::Div: case Opcode::Rem:
        case Opcode::And: case Opcode::Or: case Opcode::Xor: case Opcode::Shl: case Opcode::Shr:
        case Opcode::Min: case Opcode::Max:
          r = binary(t, in);
          break;
        case Opcode::Neg:
          if (in.type == IRType::F32) r.f = -F(t, s[0]);
          else if (in.type == IRType::I64) r.l = (int64_t)(0 - (uint64_t)L(t, s[0]));
          else r.i = wrapI32(-(int64_t)I(t, s[0]));
          break;
        case Opcode::Abs:
          if (in.type == IRType::F32) r.f = std::fabs(F(t, s[0]));
          else r.i = I(t, s[0]) == INT_MIN ? INT_MIN : std::abs(I(t, s[0]));
          break;
        case Opcode::Not: r.p = !P(t, s[0]); break;
        case Opcode::Sqrt: r.f = std::sqrt(F(t, s[0])); break;
        case Opcode::Fma:
          if (in.type == IRType::I32) r.i = wrapI32((int64_t)(int32_t)((uint32_t)I(t, s[0]) * (uint32_t)I(t, s[1])) + I(t, s[2]));
          else r.f = std::fma(F(t, s[0]), F(t, s[1]), F(t, s[2]));
          break;
        case Opcode::CmpLT: case Opcode::CmpLE: case Opcode::CmpGT:
        case Opcode::CmpGE: case Opcode::CmpEQ: case Opcode::CmpNE:
          r.p = compare(t, in);
          break;
        case Opcode::Select: {
          const Operand& pick = P(t, s[0]) ? s[1] : s[2];
          if (in.type == IRType::F32) r.f = F(t, pick);
          else if (in.type == IRType::I64) r.l = L(t, pick);
          else r.i = I(t, pick);
          break;
        }
        case Opcode::CvtI2F: r.f = (float)I(t, s[0]); break;
        case Opcode::CvtF2I: {  // PTX cvt.rzi.s32.f32: truncate, saturate, NaN -> 0
          float x = F(t, s[0]);
          if (std::isnan(x)) r.i = 0;
          else if (x >= 2147483648.0f) r.i = INT_MAX;
          else if (x <= -2147483648.0f) r.i = INT_MIN;
          else r.i = (int32_t)x;
          break;
        }
        case Opcode::CvtI2L: r.l = I(t, s[0]); break;
        case Opcode::MulWide: r.l = (int64_t)I(t, s[0]) * (int64_t)I(t, s[1]); break;
        case Opcode::LdParam: {
          const KernelArg& a = args_[in.aux];
          r.i = a.i; r.f = a.f; r.l = (int64_t)a.ptr;
          break;
        }
        case Opcode::CvtaGlobal: r.l = L(t, s[0]); break;
        case Opcode::SReg: {
          int g = in.aux / 3, d = in.aux % 3;
          unsigned ntid[3] = {block_.x, block_.y, block_.z}, nctaid[3] = {grid_.x, grid_.y, grid_.z};
          r.i = (int32_t)(g == 0 ? t.tid[d] : g == 1 ? ntid[d] : g == 2 ? ctaid_[d] : nctaid[d]);
          break;
        }
        case Opcode::SharedAddr: r.i = (int32_t)sharedOff_[in.aux]; break;
        case Opcode::LdGlobal: {
          uint8_t* p = globalPtr(t, (uint64_t)L(t, s[0]), 4, "read");
          if (in.type == IRType::F32) std::memcpy(&r.f, p, 4); else std::memcpy(&r.i, p, 4);
          break;
        }
        case Opcode::LdShared: {
          uint8_t* p = sharedPtr(t, (uint32_t)I(t, s[0]), 4, "read");
          if (in.type == IRType::F32) std::memcpy(&r.f, p, 4); else std::memcpy(&r.i, p, 4);
          break;
        }
        case Opcode::StGlobal: case Opcode::StShared: {
          uint8_t* p = in.op == Opcode::StGlobal ? globalPtr(t, (uint64_t)L(t, s[0]), 4, "write")
                                                 : sharedPtr(t, (uint32_t)I(t, s[0]), 4, "write");
          if (in.type == IRType::F32) { float v = F(t, s[1]); std::memcpy(p, &v, 4); }
          else { int32_t v = I(t, s[1]); std::memcpy(p, &v, 4); }
          break;
        }
        case Opcode::AtomAddGlobal: case Opcode::AtomAddShared: {
          uint8_t* p = in.op == Opcode::AtomAddGlobal ? globalPtr(t, (uint64_t)L(t, s[0]), 4, "atomic")
                                                      : sharedPtr(t, (uint32_t)I(t, s[0]), 4, "atomic");
          if (in.type == IRType::F32) {
            float old; std::memcpy(&old, p, 4);
            float nv = old + F(t, s[1]); std::memcpy(p, &nv, 4);
            r.f = old;
          } else {
            int32_t old; std::memcpy(&old, p, 4);
            int32_t nv = wrapI32((int64_t)old + I(t, s[1])); std::memcpy(p, &nv, 4);
            r.i = old;
          }
          break;
        }
        case Opcode::Bar:
          t.pc++;
          t.atBarrier = true;
          return;
        case Opcode::Br:
          t.bb = in.target; t.pc = 0;
          continue;
        case Opcode::CondBr:
          t.bb = P(t, s[0]) ? in.target : in.target2; t.pc = 0;
          continue;
        case Opcode::Ret:
          t.done = true;
          return;
      }
      if (write) t.regs[in.dst] = r;
      t.pc++;
    }
  }

  Val binary(const Thread& t, const Instr& in) {
    Val r;
    const Operand &a = in.src[0], &b = in.src[1];
    if (in.type == IRType::F32) {
      float x = F(t, a), y = F(t, b);
      switch (in.op) {
        case Opcode::Add: r.f = x + y; break;
        case Opcode::Sub: r.f = x - y; break;
        case Opcode::Mul: r.f = x * y; break;
        case Opcode::Div: r.f = x / y; break;
        case Opcode::Min: r.f = std::fmin(x, y); break;
        case Opcode::Max: r.f = std::fmax(x, y); break;
        default: throw RunError("bad float op");
      }
      return r;
    }
    if (in.type == IRType::Pred) {
      bool x = P(t, a), y = P(t, b);
      r.p = in.op == Opcode::And ? (x && y) : in.op == Opcode::Or ? (x || y) : (x != y);
      return r;
    }
    if (in.type == IRType::I64) {
      uint64_t x = (uint64_t)L(t, a), y = (uint64_t)L(t, b);
      switch (in.op) {
        case Opcode::Add: r.l = (int64_t)(x + y); break;
        case Opcode::Sub: r.l = (int64_t)(x - y); break;
        case Opcode::Mul: r.l = (int64_t)(x * y); break;
        default: throw RunError("bad i64 op");
      }
      return r;
    }
    int32_t x = I(t, a), y = I(t, b);
    switch (in.op) {
      case Opcode::Add: r.i = wrapI32((int64_t)x + y); break;
      case Opcode::Sub: r.i = wrapI32((int64_t)x - y); break;
      case Opcode::Mul: r.i = (int32_t)((uint32_t)x * (uint32_t)y); break;
      case Opcode::Div:
        if (y == 0) throw RunError("integer division by zero");
        r.i = (x == INT_MIN && y == -1) ? INT_MIN : x / y;
        break;
      case Opcode::Rem:
        if (y == 0) throw RunError("integer remainder by zero");
        r.i = (x == INT_MIN && y == -1) ? 0 : x % y;
        break;
      case Opcode::And: r.i = x & y; break;
      case Opcode::Or: r.i = x | y; break;
      case Opcode::Xor: r.i = x ^ y; break;
      // PTX clamps shift amounts >= 32 (C leaves them undefined)
      case Opcode::Shl: r.i = ((uint32_t)y >= 32) ? 0 : (int32_t)((uint32_t)x << y); break;
      case Opcode::Shr: r.i = ((uint32_t)y >= 32) ? (x < 0 ? -1 : 0) : (x >> y); break;
      case Opcode::Min: r.i = x < y ? x : y; break;
      case Opcode::Max: r.i = x > y ? x : y; break;
      default: throw RunError("bad int op");
    }
    return r;
  }

  bool compare(const Thread& t, const Instr& in) {
    const Operand &a = in.src[0], &b = in.src[1];
    if (in.type == IRType::F32) {
      float x = F(t, a), y = F(t, b);
      switch (in.op) {
        case Opcode::CmpLT: return x < y;
        case Opcode::CmpLE: return x <= y;
        case Opcode::CmpGT: return x > y;
        case Opcode::CmpGE: return x >= y;
        case Opcode::CmpEQ: return x == y;
        default: return x != y;  // true for NaN, like C (PTX setp.neu)
      }
    }
    int64_t x = in.type == IRType::I64 ? L(t, a) : I(t, a);
    int64_t y = in.type == IRType::I64 ? L(t, b) : I(t, b);
    switch (in.op) {
      case Opcode::CmpLT: return x < y;
      case Opcode::CmpLE: return x <= y;
      case Opcode::CmpGT: return x > y;
      case Opcode::CmpGE: return x >= y;
      case Opcode::CmpEQ: return x == y;
      default: return x != y;
    }
  }
};

}  // namespace

RunStats interpret(const Function& fn, DeviceMemory& mem, Dim3 grid, Dim3 block,
                   const std::vector<KernelArg>& args) {
  return Interp(fn, mem, grid, block, args).run();
}
