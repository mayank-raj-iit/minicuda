// ir.cpp - helper functions for the IR and a human readable printer.
#include "ir.hpp"

#include <sstream>

const char* opName(Opcode op) {
  switch (op) {
    case Opcode::Mov: return "mov";
    case Opcode::Add: return "add";
    case Opcode::Sub: return "sub";
    case Opcode::Mul: return "mul";
    case Opcode::Div: return "div";
    case Opcode::Rem: return "rem";
    case Opcode::Neg: return "neg";
    case Opcode::And: return "and";
    case Opcode::Or: return "or";
    case Opcode::Xor: return "xor";
    case Opcode::Shl: return "shl";
    case Opcode::Shr: return "shr";
    case Opcode::Not: return "not";
    case Opcode::Min: return "min";
    case Opcode::Max: return "max";
    case Opcode::Abs: return "abs";
    case Opcode::Sqrt: return "sqrt";
    case Opcode::Fma: return "fma";
    case Opcode::CmpLT: return "cmp.lt";
    case Opcode::CmpLE: return "cmp.le";
    case Opcode::CmpGT: return "cmp.gt";
    case Opcode::CmpGE: return "cmp.ge";
    case Opcode::CmpEQ: return "cmp.eq";
    case Opcode::CmpNE: return "cmp.ne";
    case Opcode::Select: return "select";
    case Opcode::CvtI2F: return "cvt.i2f";
    case Opcode::CvtF2I: return "cvt.f2i";
    case Opcode::CvtI2L: return "cvt.i2l";
    case Opcode::MulWide: return "mul.wide";
    case Opcode::LdParam: return "ld.param";
    case Opcode::CvtaGlobal: return "cvta.global";
    case Opcode::SReg: return "sreg";
    case Opcode::SharedAddr: return "shared.addr";
    case Opcode::LdGlobal: return "ld.global";
    case Opcode::LdShared: return "ld.shared";
    case Opcode::StGlobal: return "st.global";
    case Opcode::StShared: return "st.shared";
    case Opcode::AtomAddGlobal: return "atom.add.global";
    case Opcode::AtomAddShared: return "atom.add.shared";
    case Opcode::Bar: return "barrier";
    case Opcode::Br: return "br";
    case Opcode::CondBr: return "condbr";
    case Opcode::Ret: return "ret";
  }
  return "?";
}

const char* typeName(IRType t) {
  switch (t) {
    case IRType::I32: return "i32";
    case IRType::F32: return "f32";
    case IRType::I64: return "i64";
    case IRType::Pred: return "pred";
    default: return "none";
  }
}

int typeSize(IRType t) {
  switch (t) {
    case IRType::I64: return 8;
    case IRType::Pred: return 1;
    case IRType::None: return 0;
    default: return 4;
  }
}

bool isTerminator(Opcode op) { return op == Opcode::Br || op == Opcode::CondBr || op == Opcode::Ret; }

bool hasResult(Opcode op) {
  switch (op) {
    case Opcode::StGlobal: case Opcode::StShared: case Opcode::Bar:
    case Opcode::Br: case Opcode::CondBr: case Opcode::Ret:
      return false;
    default:
      return true;
  }
}

bool hasSideEffects(Opcode op) {
  switch (op) {
    case Opcode::StGlobal: case Opcode::StShared:
    case Opcode::AtomAddGlobal: case Opcode::AtomAddShared:
    case Opcode::Bar: case Opcode::Br: case Opcode::CondBr: case Opcode::Ret:
      return true;
    default:
      return false;
  }
}

bool isCompare(Opcode op) { return op >= Opcode::CmpLT && op <= Opcode::CmpNE; }

bool isCommutative(Opcode op) {
  switch (op) {
    case Opcode::Add: case Opcode::Mul: case Opcode::And: case Opcode::Or: case Opcode::Xor:
    case Opcode::Min: case Opcode::Max: case Opcode::CmpEQ: case Opcode::CmpNE: case Opcode::MulWide:
      return true;
    default:
      return false;
  }
}

IRType resultType(const Instr& in) {
  if (isCompare(in.op)) return IRType::Pred;
  if (in.op == Opcode::MulWide || in.op == Opcode::CvtI2L) return IRType::I64;
  return in.type;
}

std::vector<int> successors(const BasicBlock& b) {
  if (b.instrs.empty()) return {};
  const Instr& t = b.instrs.back();
  if (t.op == Opcode::Br) return {t.target};
  if (t.op == Opcode::CondBr) {
    if (t.target == t.target2) return {t.target};
    return {t.target, t.target2};
  }
  return {};
}

std::string operandStr(const Operand& o) {
  std::ostringstream os;
  switch (o.kind) {
    case Operand::Reg: os << "v" << o.reg; break;
    case Operand::ImmInt: os << o.imm; break;
    case Operand::ImmFloat: os << o.fimm << "f"; break;
    default: os << "<none>"; break;
  }
  return os.str();
}

void printFunction(const Function& f, std::ostream& os) {
  os << "kernel " << f.name << "(";
  for (size_t i = 0; i < f.params.size(); i++) {
    if (i) os << ", ";
    os << (f.params[i].isPtr ? std::string(typeName(f.params[i].elem)) + "*" : typeName(f.params[i].type))
       << " " << f.params[i].name;
  }
  os << ")\n";
  for (size_t i = 0; i < f.shared.size(); i++)
    os << "  shared " << f.shared[i].name << ": " << typeName(f.shared[i].elem) << "[" << f.shared[i].count << "]\n";
  for (size_t b = 0; b < f.blocks.size(); b++) {
    os << "bb" << b << ":";
    if (!f.blocks[b].name.empty()) os << "    ; " << f.blocks[b].name;
    os << "\n";
    for (const Instr& in : f.blocks[b].instrs) {
      os << "  ";
      if (in.dst >= 0) {
        os << "v" << in.dst << ":" << typeName(f.vregs[in.dst].type);
        if (!f.vregs[in.dst].hint.empty()) os << "(" << f.vregs[in.dst].hint << ")";
        os << " = ";
      }
      os << opName(in.op);
      if (in.type != IRType::None && !isTerminator(in.op)) os << "." << typeName(in.type);
      if (in.op == Opcode::LdParam) os << " param" << in.aux;
      if (in.op == Opcode::SReg) os << " " << sregName(in.aux);
      if (in.op == Opcode::SharedAddr) os << " @" << f.shared[in.aux].name;
      for (size_t s = 0; s < in.src.size(); s++) os << (s ? ", " : " ") << operandStr(in.src[s]);
      if (in.op == Opcode::Br) os << " bb" << in.target;
      if (in.op == Opcode::CondBr) os << ", bb" << in.target << ", bb" << in.target2;
      os << "\n";
    }
  }
}
