// ir.hpp - the Intermediate Representation (IR).
//
// The IR is "three-address code" organised into basic blocks:
//     v7:f32 = mul v3, v6
// Each value lives in a *virtual register* (v0, v1, ...) with a type. There
// is no limit on registers - exactly like PTX, where ptxas later does the real
// register allocation. A basic block is a straight list of instructions that
// ends in exactly one terminator (br / condbr / ret). The blocks plus the
// branches between them form the Control Flow Graph (CFG).
#pragma once
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

enum class IRType { None, I32, F32, I64, Pred };

struct Operand {
  enum Kind { None, Reg, ImmInt, ImmFloat } kind = None;
  int reg = -1;         // Reg: virtual register number
  long long imm = 0;    // ImmInt
  float fimm = 0;       // ImmFloat
  IRType type = IRType::None;

  static Operand R(int r, IRType t) { Operand o; o.kind = Reg; o.reg = r; o.type = t; return o; }
  static Operand I(long long v, IRType t = IRType::I32) { Operand o; o.kind = ImmInt; o.imm = v; o.type = t; return o; }
  static Operand F(float v) { Operand o; o.kind = ImmFloat; o.fimm = v; o.type = IRType::F32; return o; }
  bool isReg() const { return kind == Reg; }
  bool isImm() const { return kind == ImmInt || kind == ImmFloat; }
};

enum class Opcode {
  // value = op(src...)       (type = operand/result type)
  Mov, Add, Sub, Mul, Div, Rem, Neg, And, Or, Xor, Shl, Shr, Not,
  Min, Max, Abs, Sqrt, Fma,
  // pred = compare(a, b)     (type = type of a and b)
  CmpLT, CmpLE, CmpGT, CmpGE, CmpEQ, CmpNE,
  Select,      // dst = src0(pred) ? src1 : src2
  CvtI2F,      // f32 = (float)i32
  CvtF2I,      // i32 = (int)f32, rounds toward zero
  CvtI2L,      // i64 = (long)i32
  MulWide,     // i64 = i32 * i32 (full 64-bit product), used for byte offsets
  LdParam,     // dst = kernel parameter number `aux`
  CvtaGlobal,  // dst = generic address -> global address
  SReg,        // dst = special register `aux` (tid.x, ntid.x, ctaid.x, nctaid.x ...)
  SharedAddr,  // dst(i32) = address of shared array `aux`
  LdGlobal, LdShared,          // dst = *src0
  StGlobal, StShared,          // *src0 = src1
  AtomAddGlobal, AtomAddShared,// dst = old *src0; *src0 += src1  (atomically)
  Bar,                         // __syncthreads()
  Br, CondBr, Ret              // terminators
};

struct Instr {
  Opcode op = Opcode::Mov;
  IRType type = IRType::None;
  int dst = -1;                 // destination register, -1 if none
  std::vector<Operand> src;
  int target = -1, target2 = -1;// Br: target.  CondBr: target if true, target2 if false
  int aux = 0;                  // LdParam index / SReg id / SharedAddr index
  int line = 0, col = 0;        // source position, for .loc debug line info
};

struct BasicBlock {
  std::string name;             // human readable hint ("for.cond", "if.then", ...)
  std::vector<Instr> instrs;
};

struct ParamInfo {
  std::string name;
  IRType type;                  // I32 / F32 / I64 (pointer)
  bool isPtr = false;
  IRType elem = IRType::None;   // pointer element type
};

struct SharedVar {
  std::string name;
  IRType elem;
  long long count;
};

struct VRegInfo {
  IRType type;
  std::string hint;             // variable name, if it holds a source variable
};

struct Function {
  std::string name;
  std::vector<ParamInfo> params;
  std::vector<VRegInfo> vregs;
  std::vector<BasicBlock> blocks;  // blocks[0] is the entry; vector order = layout order
  std::vector<SharedVar> shared;

  int newReg(IRType t, const std::string& hint = "") {
    vregs.push_back({t, hint});
    return (int)vregs.size() - 1;
  }
  size_t instrCount() const {
    size_t n = 0;
    for (auto& b : blocks) n += b.instrs.size();
    return n;
  }
};

// ---- special register ids for SReg ----
// id = group*3 + dim, groups: 0 tid, 1 ntid, 2 ctaid, 3 nctaid
inline const char* sregName(int id) {
  static const char* n[] = {"%tid.x", "%tid.y", "%tid.z", "%ntid.x", "%ntid.y", "%ntid.z",
                            "%ctaid.x", "%ctaid.y", "%ctaid.z", "%nctaid.x", "%nctaid.y", "%nctaid.z"};
  return n[id];
}

// ---- helpers about opcodes (defined in ir.cpp) ----
const char* opName(Opcode op);
const char* typeName(IRType t);
bool hasResult(Opcode op);
bool isTerminator(Opcode op);
bool hasSideEffects(Opcode op);       // must never be deleted (stores, atomics, barriers, branches)
bool isCompare(Opcode op);
bool isCommutative(Opcode op);
IRType resultType(const Instr& in);   // type of dst
std::vector<int> successors(const BasicBlock& b);
int typeSize(IRType t);               // bytes

void printFunction(const Function& f, std::ostream& os);
std::string operandStr(const Operand& o);
