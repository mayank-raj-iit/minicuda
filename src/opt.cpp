// opt.cpp - the optimizer.
//
// Passes:
//  1. simplifyCFG     - constant branches become jumps, unreachable blocks are
//                       deleted, a block with a single predecessor is merged
//                       into it, and blocks that only jump are bypassed.
//  2. LVN             - "local value numbering" inside each basic block:
//                       constant folding, constant/copy propagation, algebraic
//                       simplification (x*1, x+0, x*8 -> x<<3), common
//                       subexpression elimination, and redundant load removal.
//  3. DCE             - liveness analysis over the CFG, then delete
//                       instructions whose results are never used.
//     Global copy/constant propagation - "int i = expr;" leaves a copy i = t;
//                       when both are assigned exactly once, uses of i become t
//                       (and uses of a variable assigned a constant once become that constant).
//  4. LICM            - find natural loops with dominators, and hoist
//                       computations that give the same value on every
//                       iteration into a preheader block.
//  5. FMA contraction - a*b + c  ->  fma(a, b, c), like nvcc --fmad=true.
//  6. copy coalescing - t = a + b; x = t  ->  x = a + b
#include "opt.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_map>

// ======================================================================
// Analyses
// ======================================================================

std::vector<std::vector<int>> computePreds(const Function& f) {
  std::vector<std::vector<int>> preds(f.blocks.size());
  for (size_t b = 0; b < f.blocks.size(); b++)
    for (int s : successors(f.blocks[b])) preds[s].push_back((int)b);
  return preds;
}

// Classic backward dataflow:  out[b] = U in[succ],  in[b] = use[b] U (out[b] - def[b])
Liveness computeLiveness(const Function& f) {
  const size_t nb = f.blocks.size(), nr = f.vregs.size();
  std::vector<std::vector<char>> use(nb, std::vector<char>(nr, 0)), def(nb, std::vector<char>(nr, 0));
  for (size_t b = 0; b < nb; b++)
    for (const Instr& in : f.blocks[b].instrs) {
      for (const Operand& o : in.src)
        if (o.isReg() && !def[b][o.reg]) use[b][o.reg] = 1;
      if (in.dst >= 0) def[b][in.dst] = 1;
    }
  Liveness lv;
  lv.in.assign(nb, std::vector<char>(nr, 0));
  lv.out.assign(nb, std::vector<char>(nr, 0));
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t bi = nb; bi-- > 0;) {
      std::vector<char> out(nr, 0);
      for (int s : successors(f.blocks[bi]))
        for (size_t r = 0; r < nr; r++) out[r] |= lv.in[s][r];
      std::vector<char> in(nr, 0);
      for (size_t r = 0; r < nr; r++) in[r] = use[bi][r] | (out[r] & !def[bi][r]);
      if (in != lv.in[bi] || out != lv.out[bi]) {
        lv.in[bi] = std::move(in);
        lv.out[bi] = std::move(out);
        changed = true;
      }
    }
  }
  return lv;
}

namespace {

// Renumber blocks after deleting/inserting; newId[b] == -1 means "deleted".
void remapBlocks(Function& f, const std::vector<int>& newId, std::vector<BasicBlock> newBlocks) {
  for (auto& blk : newBlocks)
    for (auto& in : blk.instrs) {
      if (in.op == Opcode::Br || in.op == Opcode::CondBr) in.target = newId[in.target];
      if (in.op == Opcode::CondBr) in.target2 = newId[in.target2];
    }
  f.blocks = std::move(newBlocks);
}

bool removeUnreachable(Function& f) {
  std::vector<char> seen(f.blocks.size(), 0);
  std::vector<int> stack = {0};
  seen[0] = 1;
  while (!stack.empty()) {
    int b = stack.back();
    stack.pop_back();
    for (int s : successors(f.blocks[b]))
      if (!seen[s]) { seen[s] = 1; stack.push_back(s); }
  }
  if (std::all_of(seen.begin(), seen.end(), [](char c) { return c; })) return false;
  std::vector<int> newId(f.blocks.size(), -1);
  std::vector<BasicBlock> kept;
  for (size_t b = 0; b < f.blocks.size(); b++)
    if (seen[b]) { newId[b] = (int)kept.size(); kept.push_back(std::move(f.blocks[b])); }
  remapBlocks(f, newId, std::move(kept));
  return true;
}

// Insert an empty block at layout position `pos`; returns its id (== pos).
int insertBlock(Function& f, int pos, const std::string& name) {
  std::vector<int> newId(f.blocks.size());
  std::vector<BasicBlock> nb;
  for (size_t b = 0; b < f.blocks.size(); b++) {
    if ((int)b == pos) nb.push_back(BasicBlock{name, {}});
    newId[b] = (int)nb.size();
    nb.push_back(std::move(f.blocks[b]));
  }
  if (pos >= (int)f.blocks.size()) nb.push_back(BasicBlock{name, {}});
  remapBlocks(f, newId, std::move(nb));
  return pos;
}

void retarget(Instr& term, int from, int to) {
  if ((term.op == Opcode::Br || term.op == Opcode::CondBr) && term.target == from) term.target = to;
  if (term.op == Opcode::CondBr && term.target2 == from) term.target2 = to;
}

// ---------- constant evaluation (must match interp.cpp / the GPU exactly) ----------
int32_t wrap32(int64_t v) { return (int32_t)(uint32_t)(uint64_t)v; }

bool isPow2(long long v) { return v > 0 && (v & (v - 1)) == 0; }
int log2i(long long v) { int k = 0; while ((1LL << k) < v) k++; return k; }

// Try to compute the result of `in` given constant operand values. Returns
// nullopt when the operation cannot (or must not) be folded.
std::optional<Operand> fold(const Instr& in, const std::vector<Operand>& v) {
  auto I = [&](int k) { return (int32_t)v[k].imm; };
  auto L = [&](int k) { return (int64_t)v[k].imm; };
  auto F = [&](int k) { return v[k].fimm; };
  auto P = [&](int k) { return v[k].imm != 0; };
  const IRType t = in.type;
  switch (in.op) {
    case Opcode::Add: case Opcode::Sub: case Opcode::Mul: case Opcode::Div: case Opcode::Rem:
    case Opcode::And: case Opcode::Or: case Opcode::Xor: case Opcode::Shl: case Opcode::Shr:
    case Opcode::Min: case Opcode::Max:
      if (t == IRType::F32) {
        float a = F(0), b = F(1), r;
        switch (in.op) {
          case Opcode::Add: r = a + b; break;
          case Opcode::Sub: r = a - b; break;
          case Opcode::Mul: r = a * b; break;
          case Opcode::Div: r = a / b; break;
          case Opcode::Min: r = std::fmin(a, b); break;
          case Opcode::Max: r = std::fmax(a, b); break;
          default: return std::nullopt;
        }
        return Operand::F(r);
      }
      if (t == IRType::Pred) {
        bool a = P(0), b = P(1);
        bool r = in.op == Opcode::And ? (a && b) : in.op == Opcode::Or ? (a || b) : (a != b);
        return Operand::I(r, IRType::Pred);
      }
      if (t == IRType::I64) {
        uint64_t a = (uint64_t)L(0), b = (uint64_t)L(1);
        if (in.op == Opcode::Add) return Operand::I((long long)(a + b), IRType::I64);
        if (in.op == Opcode::Sub) return Operand::I((long long)(a - b), IRType::I64);
        if (in.op == Opcode::Mul) return Operand::I((long long)(a * b), IRType::I64);
        return std::nullopt;
      } else {
        int32_t a = I(0), b = I(1), r;
        switch (in.op) {
          case Opcode::Add: r = wrap32((int64_t)a + b); break;
          case Opcode::Sub: r = wrap32((int64_t)a - b); break;
          case Opcode::Mul: r = (int32_t)((uint32_t)a * (uint32_t)b); break;
          case Opcode::Div: if (b == 0) return std::nullopt; r = (a == INT_MIN && b == -1) ? INT_MIN : a / b; break;
          case Opcode::Rem: if (b == 0) return std::nullopt; r = (a == INT_MIN && b == -1) ? 0 : a % b; break;
          case Opcode::And: r = a & b; break;
          case Opcode::Or: r = a | b; break;
          case Opcode::Xor: r = a ^ b; break;
          case Opcode::Shl: r = ((uint32_t)b >= 32) ? 0 : (int32_t)((uint32_t)a << b); break;
          case Opcode::Shr: r = ((uint32_t)b >= 32) ? (a < 0 ? -1 : 0) : (a >> b); break;
          case Opcode::Min: r = a < b ? a : b; break;
          case Opcode::Max: r = a > b ? a : b; break;
          default: return std::nullopt;
        }
        return Operand::I(r);
      }
    case Opcode::Neg:
      if (t == IRType::F32) return Operand::F(-F(0));
      if (t == IRType::I32) return Operand::I(wrap32(-(int64_t)I(0)));
      return std::nullopt;
    case Opcode::Abs:
      if (t == IRType::F32) return Operand::F(std::fabs(F(0)));
      return Operand::I(I(0) == INT_MIN ? INT_MIN : std::abs(I(0)));
    case Opcode::Not: return Operand::I(!P(0), IRType::Pred);
    case Opcode::Sqrt: return Operand::F(std::sqrt(F(0)));
    case Opcode::Fma:
      if (t == IRType::I32) return Operand::I(wrap32((int64_t)(int32_t)((uint32_t)I(0) * (uint32_t)I(1)) + I(2)));
      return Operand::F(std::fma(F(0), F(1), F(2)));
    case Opcode::CmpLT: case Opcode::CmpLE: case Opcode::CmpGT:
    case Opcode::CmpGE: case Opcode::CmpEQ: case Opcode::CmpNE: {
      bool r;
      if (t == IRType::F32) {
        float a = F(0), b = F(1);
        r = in.op == Opcode::CmpLT ? a < b : in.op == Opcode::CmpLE ? a <= b : in.op == Opcode::CmpGT ? a > b
          : in.op == Opcode::CmpGE ? a >= b : in.op == Opcode::CmpEQ ? a == b : a != b;
      } else {
        int64_t a = v[0].imm, b = v[1].imm;
        if (t == IRType::I32) { a = (int32_t)a; b = (int32_t)b; }
        r = in.op == Opcode::CmpLT ? a < b : in.op == Opcode::CmpLE ? a <= b : in.op == Opcode::CmpGT ? a > b
          : in.op == Opcode::CmpGE ? a >= b : in.op == Opcode::CmpEQ ? a == b : a != b;
      }
      return Operand::I(r, IRType::Pred);
    }
    case Opcode::Select: {
      Operand pick = P(0) ? v[1] : v[2];
      pick.type = t;
      return pick;
    }
    case Opcode::CvtI2F: return Operand::F((float)I(0));
    case Opcode::CvtF2I: {
      float x = F(0);
      if (std::isnan(x)) return Operand::I(0);
      if (x >= 2147483648.0f) return Operand::I(INT_MAX);
      if (x <= -2147483648.0f) return Operand::I(INT_MIN);
      return Operand::I((int32_t)x);
    }
    case Opcode::CvtI2L: return Operand::I((int64_t)I(0), IRType::I64);
    case Opcode::MulWide: return Operand::I((int64_t)I(0) * (int64_t)I(1), IRType::I64);
    default: return std::nullopt;
  }
}

bool sameOperand(const Operand& a, const Operand& b) {
  if (a.kind != b.kind) return false;
  if (a.kind == Operand::Reg) return a.reg == b.reg;
  if (a.kind == Operand::ImmInt) return a.imm == b.imm;
  if (a.kind == Operand::ImmFloat) return std::memcmp(&a.fimm, &b.fimm, 4) == 0;
  return true;
}
bool isIntImm(const Operand& o, long long v) { return o.kind == Operand::ImmInt && o.imm == v; }
bool isFloatImm(const Operand& o, float v) { return o.kind == Operand::ImmFloat && o.fimm == v && !std::signbit(o.fimm) == !std::signbit(v); }

void makeMov(Instr& in, Operand src) {
  IRType rt = resultType(in);
  in.op = Opcode::Mov;
  in.type = rt;
  src.type = rt;
  in.src = {src};
  in.aux = 0;
}

// Algebraic identities. Returns true if `in` was rewritten.
bool simplify(Instr& in) {
  auto& s = in.src;
  // put constants second for commutative ops:  4 * x  ->  x * 4
  if (isCommutative(in.op) && s.size() == 2 && s[0].isImm() && s[1].isReg()) {
    std::swap(s[0], s[1]);
    return simplify(in), true;
  }
  const bool isInt = in.type == IRType::I32 || in.type == IRType::I64;
  switch (in.op) {
    case Opcode::Add:
      if (isInt && isIntImm(s[1], 0)) { makeMov(in, s[0]); return true; }
      break;
    case Opcode::Sub:
      if (isInt && isIntImm(s[1], 0)) { makeMov(in, s[0]); return true; }
      if (in.type == IRType::F32 && isFloatImm(s[1], 0.0f)) { makeMov(in, s[0]); return true; }
      if (isInt && s[0].isReg() && sameOperand(s[0], s[1])) { makeMov(in, Operand::I(0, in.type)); return true; }
      break;
    case Opcode::Mul:
      if (isInt && isIntImm(s[1], 1)) { makeMov(in, s[0]); return true; }
      if (isInt && isIntImm(s[1], 0)) { makeMov(in, Operand::I(0, in.type)); return true; }
      if (in.type == IRType::F32 && isFloatImm(s[1], 1.0f)) { makeMov(in, s[0]); return true; }
      if (in.type == IRType::I32 && s[1].kind == Operand::ImmInt && isPow2(s[1].imm)) {
        // strength reduction: x * 8  ->  x << 3  (same result, cheaper)
        in.op = Opcode::Shl;
        s[1] = Operand::I(log2i(s[1].imm));
        return true;
      }
      break;
    case Opcode::Div:
      if (isInt && isIntImm(s[1], 1)) { makeMov(in, s[0]); return true; }
      if (in.type == IRType::F32 && isFloatImm(s[1], 1.0f)) { makeMov(in, s[0]); return true; }
      break;
    case Opcode::And:
      if (in.type == IRType::I32 && isIntImm(s[1], 0)) { makeMov(in, Operand::I(0)); return true; }
      if (in.type == IRType::I32 && isIntImm(s[1], -1)) { makeMov(in, s[0]); return true; }
      if (s[0].isReg() && sameOperand(s[0], s[1])) { makeMov(in, s[0]); return true; }
      break;
    case Opcode::Or:
      if (in.type == IRType::I32 && isIntImm(s[1], 0)) { makeMov(in, s[0]); return true; }
      if (s[0].isReg() && sameOperand(s[0], s[1])) { makeMov(in, s[0]); return true; }
      break;
    case Opcode::Xor:
      if (in.type == IRType::I32 && isIntImm(s[1], 0)) { makeMov(in, s[0]); return true; }
      if (in.type == IRType::I32 && s[0].isReg() && sameOperand(s[0], s[1])) { makeMov(in, Operand::I(0)); return true; }
      break;
    case Opcode::Shl: case Opcode::Shr:
      if (isIntImm(s[1], 0)) { makeMov(in, s[0]); return true; }
      break;
    case Opcode::Min: case Opcode::Max:
      if (s[0].isReg() && sameOperand(s[0], s[1])) { makeMov(in, s[0]); return true; }
      break;
    case Opcode::Select:
      if (s[0].isImm()) { makeMov(in, s[0].imm ? s[1] : s[2]); return true; }
      if (sameOperand(s[1], s[2])) { makeMov(in, s[1]); return true; }
      break;
    default: break;
  }
  return false;
}

bool isLoad(Opcode op) { return op == Opcode::LdGlobal || op == Opcode::LdShared; }

// Operations whose result depends only on their operands (safe to reuse or move).
bool isPureValue(Opcode op) {
  if (!hasResult(op) || hasSideEffects(op) || isLoad(op)) return false;
  return op != Opcode::Mov;
}

}  // namespace

// ======================================================================
// Pass 1: CFG simplification
// ======================================================================
bool simplifyCFG(Function& f) {
  bool changed = false;
  for (int round = 0; round < 100; round++) {
    bool again = false;
    // (a) condbr on a constant, or with both targets equal -> br
    for (auto& blk : f.blocks) {
      Instr& t = blk.instrs.back();
      if (t.op != Opcode::CondBr) continue;
      if (t.src[0].isImm() || t.target == t.target2) {
        int to = (t.target == t.target2 || t.src[0].imm) ? t.target : t.target2;
        t.op = Opcode::Br;
        t.target = to;
        t.target2 = -1;
        t.src.clear();
        again = true;
      }
    }
    // (b) bypass blocks that contain nothing but "br X"
    for (size_t b = 1; b < f.blocks.size(); b++) {
      const auto& ins = f.blocks[b].instrs;
      if (ins.size() != 1 || ins[0].op != Opcode::Br || ins[0].target == (int)b) continue;
      int to = ins[0].target;
      for (size_t p = 0; p < f.blocks.size(); p++) {
        if (p == b) continue;
        Instr& t = f.blocks[p].instrs.back();
        Instr before = t;
        retarget(t, (int)b, to);
        if (t.target != before.target || t.target2 != before.target2) again = true;
      }
    }
    // (c) delete blocks nobody can reach
    if (removeUnreachable(f)) again = true;
    // (d) merge B into A when A always jumps to B and B has no other predecessor
    auto preds = computePreds(f);
    for (size_t a = 0; a < f.blocks.size(); a++) {
      Instr& t = f.blocks[a].instrs.back();
      if (t.op != Opcode::Br) continue;
      int b = t.target;
      if (b == (int)a || b == 0 || preds[b].size() != 1) continue;
      f.blocks[a].instrs.pop_back();
      auto& src = f.blocks[b].instrs;
      f.blocks[a].instrs.insert(f.blocks[a].instrs.end(), src.begin(), src.end());
      src.clear();
      src.push_back(Instr{});            // placeholder terminator; block is now unreachable
      src.back().op = Opcode::Ret;
      removeUnreachable(f);
      again = true;
      break;  // indices changed; recompute
    }
    if (!again) break;
    changed = true;
  }
  return changed;
}

// ======================================================================
// Pass 2: local value numbering
// ======================================================================
bool localValueNumbering(Function& f) {
  bool changed = false;
  for (auto& blk : f.blocks) {
    std::unordered_map<int, int> regVN;                 // register -> value number
    std::unordered_map<int, Operand> vnConst;           // value number -> constant
    std::unordered_map<int, int> vnHome;                // value number -> a register holding it now
    std::map<std::tuple<int, int, long long>, int> constVN;
    using Key = std::tuple<int, int, int, int, int, int, int>;
    std::map<Key, int> exprVN;
    int nextVN = 0, memEpoch = 0;

    auto vnOfConst = [&](const Operand& o) {
      long long bits = o.imm;
      if (o.kind == Operand::ImmFloat) { int32_t b; std::memcpy(&b, &o.fimm, 4); bits = b; }
      auto key = std::make_tuple((int)o.kind, (int)o.type, bits);
      auto it = constVN.find(key);
      if (it != constVN.end()) return it->second;
      int vn = nextVN++;
      constVN[key] = vn;
      vnConst[vn] = o;
      return vn;
    };
    auto vnOf = [&](const Operand& o) {
      if (o.isImm()) return vnOfConst(o);
      auto it = regVN.find(o.reg);
      if (it != regVN.end()) return it->second;
      int vn = nextVN++;               // value the register had when the block started
      regVN[o.reg] = vn;
      vnHome[vn] = o.reg;
      return vn;
    };
    auto constOf = [&](const Operand& o) -> std::optional<Operand> {
      if (o.isImm()) return o;
      auto c = vnConst.find(vnOf(o));
      if (c == vnConst.end()) return std::nullopt;
      return c->second;
    };

    std::vector<Instr> out;
    for (Instr in : blk.instrs) {
      // 1. rewrite operands: constants become immediates, copies use the original
      for (size_t k = 0; k < in.src.size(); k++) {
        Operand& o = in.src[k];
        if (!o.isReg()) continue;
        int vn = vnOf(o);
        auto c = vnConst.find(vn);
        // predicates may only become immediates where they get folded away right now
        bool predOk = o.type != IRType::Pred || in.op == Opcode::CondBr || (in.op == Opcode::Select && k == 0);
        if (c != vnConst.end() && predOk) {
          Operand imm = c->second;
          imm.type = o.type;
          o = imm;
          changed = true;
          continue;
        }
        auto h = vnHome.find(vn);
        if (h != vnHome.end() && h->second != o.reg) {
          o = Operand::R(h->second, o.type);
          changed = true;
        }
      }

      if (isTerminator(in.op)) {
        out.push_back(in);
        continue;
      }

      // 2. constant folding
      std::optional<Operand> predConst;
      if (in.dst >= 0 && isPureValue(in.op) && in.op != Opcode::LdParam && in.op != Opcode::SReg &&
          in.op != Opcode::SharedAddr && in.op != Opcode::CvtaGlobal) {
        std::vector<Operand> vals;
        bool allConst = true;
        for (auto& o : in.src) {
          auto c = constOf(o);
          if (!c) { allConst = false; break; }
          vals.push_back(*c);
        }
        if (allConst) {
          if (auto r = fold(in, vals)) {
            if (resultType(in) == IRType::Pred) predConst = r;  // keep the setp, remember the value
            else { makeMov(in, *r); changed = true; }
          }
        }
      }

      // 3. algebraic simplification / strength reduction
      if (in.op != Opcode::Mov && simplify(in)) changed = true;

      // 4. value-number the result
      if (in.dst < 0) {  // store or barrier
        memEpoch++;
        if (in.op == Opcode::StGlobal || in.op == Opcode::StShared) {
          // a later load from the same address in this block can reuse the stored value
          Opcode ld = in.op == Opcode::StGlobal ? Opcode::LdGlobal : Opcode::LdShared;
          int valVN = vnOf(in.src[1]);
          if (in.src[1].isReg() && !vnHome.count(valVN)) vnHome[valVN] = in.src[1].reg;
          exprVN[Key((int)ld, (int)in.type, 0, vnOf(in.src[0]), -1, -1, memEpoch)] = valVN;
        }
        out.push_back(in);
        continue;
      }

      int vn;
      if (in.op == Opcode::Mov) {
        if (in.src[0].isReg() && in.src[0].reg == in.dst) { changed = true; continue; }  // x = x
        vn = vnOf(in.src[0]);
      } else if (in.op == Opcode::AtomAddGlobal || in.op == Opcode::AtomAddShared) {
        memEpoch++;
        vn = nextVN++;
      } else {
        int a = in.src.size() > 0 ? vnOf(in.src[0]) : -1;
        int b = in.src.size() > 1 ? vnOf(in.src[1]) : -1;
        int c = in.src.size() > 2 ? vnOf(in.src[2]) : -1;
        if (isCommutative(in.op) && a > b) std::swap(a, b);
        Key key((int)in.op, (int)in.type, in.aux, a, b, c, isLoad(in.op) ? memEpoch : -1);
        auto it = exprVN.find(key);
        vn = -1;
        if (it != exprVN.end()) {
          int evn = it->second;
          auto cst = vnConst.find(evn);
          auto h = vnHome.find(evn);
          if (cst != vnConst.end() && resultType(in) != IRType::Pred) {
            makeMov(in, cst->second);
            vn = evn;
            changed = true;
          } else if (h != vnHome.end()) {  // common subexpression: reuse the earlier result
            makeMov(in, Operand::R(h->second, resultType(in)));
            vn = evn;
            changed = true;
          }
        }
        if (vn < 0) {
          vn = nextVN++;
          exprVN[key] = vn;
        }
        if (predConst) vnConst[vn] = *predConst;
      }

      // 5. the destination now holds `vn`; it no longer holds its old value
      auto old = regVN.find(in.dst);
      if (old != regVN.end()) {
        auto h = vnHome.find(old->second);
        if (h != vnHome.end() && h->second == in.dst) vnHome.erase(h);
      }
      regVN[in.dst] = vn;
      if (!vnHome.count(vn)) vnHome[vn] = in.dst;
      out.push_back(in);
    }
    blk.instrs = std::move(out);
  }
  return changed;
}

// ======================================================================
// Pass 3: dead code elimination
// ======================================================================
bool deadCodeElimination(Function& f) {
  bool changed = false;
  while (true) {
    Liveness lv = computeLiveness(f);
    bool removed = false;
    for (size_t b = 0; b < f.blocks.size(); b++) {
      std::vector<char> live = lv.out[b];
      auto& ins = f.blocks[b].instrs;
      std::vector<char> keep(ins.size(), 1);
      for (size_t i = ins.size(); i-- > 0;) {
        const Instr& in = ins[i];
        bool selfCopy = in.op == Opcode::Mov && in.src[0].isReg() && in.src[0].reg == in.dst;
        bool needed = hasSideEffects(in.op) || (in.dst >= 0 && live[in.dst] && !selfCopy);
        if (!needed) { keep[i] = 0; removed = true; continue; }
        if (in.dst >= 0) live[in.dst] = 0;
        for (const Operand& o : in.src)
          if (o.isReg()) live[o.reg] = 1;
      }
      std::vector<Instr> kept;
      for (size_t i = 0; i < ins.size(); i++)
        if (keep[i]) kept.push_back(std::move(ins[i]));
      ins = std::move(kept);
    }
    if (!removed) return changed;
    changed = true;
  }
}

// ======================================================================
// Pass 3b: global copy propagation
// ======================================================================
namespace {
std::vector<std::vector<char>> dominators(const Function& f, const std::vector<std::vector<int>>& preds);
}

bool globalCopyPropagation(Function& f) {
  removeUnreachable(f);
  std::vector<int> defs(f.vregs.size(), 0);
  for (auto& blk : f.blocks)
    for (auto& in : blk.instrs)
      if (in.dst >= 0) defs[in.dst]++;
  auto dom = dominators(f, computePreds(f));
  bool changed = false;
  for (size_t b = 0; b < f.blocks.size(); b++) {
    for (size_t i = 0; i < f.blocks[b].instrs.size(); i++) {
      const Instr mv = f.blocks[b].instrs[i];
      if (mv.op != Opcode::Mov) continue;
      const bool isConst = mv.src[0].isImm();   // x = 5 (assigned once): uses of x become 5
      if (isConst && f.vregs[mv.dst].type == IRType::Pred) continue;
      if (!isConst && !mv.src[0].isReg()) continue;
      int d = mv.dst, s = isConst ? -1 : mv.src[0].reg;
      if (d == s || defs[d] != 1 || (!isConst && defs[s] != 1)) continue;
      // every use of d must come after the copy (the copy dominates the use)
      bool ok = true;
      for (size_t ub = 0; ub < f.blocks.size() && ok; ub++)
        for (size_t ui = 0; ui < f.blocks[ub].instrs.size() && ok; ui++)
          for (const Operand& o : f.blocks[ub].instrs[ui].src)
            if (o.isReg() && o.reg == d) {
              bool dominated = ub == b ? ui > i : dom[ub][b] != 0;
              if (!dominated) ok = false;
            }
      if (!ok) continue;
      for (auto& blk : f.blocks)
        for (auto& in : blk.instrs)
          for (auto& o : in.src)
            if (o.isReg() && o.reg == d) {
              if (isConst) { IRType t = o.type; o = mv.src[0]; o.type = t; }
              else o.reg = s;
              changed = true;
            }
    }
  }
  return changed;
}

// ======================================================================
// Pass 3c: copy coalescing   t = op(...); ...; x = t   ->   x = op(...)
// (removes the extra mov that "sum += v" or "i++" would otherwise leave)
// ======================================================================
bool coalesceCopies(Function& f) {
  std::vector<int> uses(f.vregs.size(), 0), defs(f.vregs.size(), 0);
  for (auto& blk : f.blocks)
    for (auto& in : blk.instrs) {
      if (in.dst >= 0) defs[in.dst]++;
      for (auto& o : in.src) if (o.isReg()) uses[o.reg]++;
    }
  bool changed = false;
  for (auto& blk : f.blocks) {
    auto& ins = blk.instrs;
    for (size_t m = 0; m < ins.size(); m++) {
      if (ins[m].op != Opcode::Mov || !ins[m].src[0].isReg()) continue;
      int x = ins[m].dst, t = ins[m].src[0].reg;
      if (x == t || uses[t] != 1 || defs[t] != 1 || f.vregs[x].type != f.vregs[t].type) continue;
      int j = (int)m - 1;
      while (j >= 0 && ins[j].dst != t) j--;
      if (j < 0) continue;
      bool clash = false;  // x must not be read or written between the def of t and the copy
      for (size_t k = j + 1; k < m && !clash; k++) {
        if (ins[k].dst == x) clash = true;
        for (auto& o : ins[k].src) if (o.isReg() && o.reg == x) clash = true;
      }
      if (clash) continue;
      ins[j].dst = x;
      ins.erase(ins.begin() + m);
      defs[t] = 0;
      m--;
      changed = true;
    }
  }
  return changed;
}

// ======================================================================
// Pass 4: loop-invariant code motion
// ======================================================================
namespace {

struct Loop {
  int header;
  std::set<int> blocks;
};

// dom[b][d] == 1  <=>  every path from entry to b passes through d
std::vector<std::vector<char>> dominators(const Function& f, const std::vector<std::vector<int>>& preds) {
  size_t n = f.blocks.size();
  std::vector<std::vector<char>> dom(n, std::vector<char>(n, 1));
  dom[0].assign(n, 0);
  dom[0][0] = 1;
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t b = 1; b < n; b++) {
      std::vector<char> d(n, 1);
      for (int p : preds[b])
        for (size_t k = 0; k < n; k++) d[k] &= dom[p][k];
      if (preds[b].empty()) d.assign(n, 0);
      d[b] = 1;
      if (d != dom[b]) { dom[b] = std::move(d); changed = true; }
    }
  }
  return dom;
}

std::vector<Loop> findLoops(const Function& f) {
  auto preds = computePreds(f);
  auto dom = dominators(f, preds);
  std::map<int, std::set<int>> byHeader;
  for (size_t t = 0; t < f.blocks.size(); t++)
    for (int h : successors(f.blocks[t])) {
      if (!dom[t][h]) continue;  // t -> h is a back edge when h dominates t
      std::set<int>& body = byHeader[h];
      body.insert(h);
      std::vector<int> stack;
      if (body.insert((int)t).second) stack.push_back((int)t);
      while (!stack.empty()) {   // everything that reaches t without passing h
        int x = stack.back();
        stack.pop_back();
        for (int p : preds[x])
          if (body.insert(p).second) stack.push_back(p);
      }
    }
  std::vector<Loop> loops;
  for (auto& [h, body] : byHeader) loops.push_back({h, body});
  std::sort(loops.begin(), loops.end(), [](const Loop& a, const Loop& b) { return a.blocks.size() < b.blocks.size(); });
  return loops;
}

bool hoistable(Opcode op) {
  if (!isPureValue(op) && op != Opcode::Mov) return false;
  return op != Opcode::Div && op != Opcode::Rem;  // never speculate a division
}

bool hoistFromLoop(Function& f, const Loop& L) {
  if (L.header == 0) return false;
  Liveness lv = computeLiveness(f);
  std::vector<int> defsInLoop(f.vregs.size(), 0);
  for (int b : L.blocks)
    for (const Instr& in : f.blocks[b].instrs)
      if (in.dst >= 0) defsInLoop[in.dst]++;
  std::vector<int> exits;
  for (int b : L.blocks)
    for (int s : successors(f.blocks[b]))
      if (!L.blocks.count(s)) exits.push_back(s);

  std::vector<std::vector<char>> moved(f.blocks.size());
  for (int b : L.blocks) moved[b].assign(f.blocks[b].instrs.size(), 0);
  std::vector<Instr> hoisted;
  bool progress = true;
  while (progress) {
    progress = false;
    for (int b : L.blocks) {
      auto& ins = f.blocks[b].instrs;
      for (size_t i = 0; i < ins.size(); i++) {
        const Instr& in = ins[i];
        if (moved[b][i] || in.dst < 0 || !hoistable(in.op)) continue;
        bool invariant = true;
        for (const Operand& o : in.src)
          if (o.isReg() && defsInLoop[o.reg] != 0) invariant = false;
        if (!invariant || defsInLoop[in.dst] != 1 || lv.in[L.header][in.dst]) continue;
        bool liveAtExit = false;
        for (int e : exits) liveAtExit |= lv.in[e][in.dst] != 0;
        if (liveAtExit) continue;
        moved[b][i] = 1;
        defsInLoop[in.dst] = 0;   // now defined outside the loop
        hoisted.push_back(in);
        progress = true;
      }
    }
  }
  if (hoisted.empty()) return false;

  for (int b : L.blocks) {
    auto& ins = f.blocks[b].instrs;
    std::vector<Instr> kept;
    for (size_t i = 0; i < ins.size(); i++)
      if (!moved[b][i]) kept.push_back(std::move(ins[i]));
    ins = std::move(kept);
  }

  // find or create the preheader: the single block that enters the loop
  auto preds = computePreds(f);
  std::vector<int> outside;
  for (int p : preds[L.header])
    if (!L.blocks.count(p)) outside.push_back(p);
  int ph;
  if (outside.size() == 1 && successors(f.blocks[outside[0]]).size() == 1) {
    ph = outside[0];
  } else {
    int header = L.header;
    ph = insertBlock(f, header, "loop.preheader");
    header++;  // shifted by the insertion
    for (int& p : outside) if (p >= ph) p++;
    for (int p : outside) retarget(f.blocks[p].instrs.back(), header, ph);
    Instr br;
    br.op = Opcode::Br;
    br.target = header;
    f.blocks[ph].instrs.push_back(br);
  }
  auto& phIns = f.blocks[ph].instrs;
  phIns.insert(phIns.end() - 1, hoisted.begin(), hoisted.end());
  return true;
}

}  // namespace

bool loopInvariantCodeMotion(Function& f) {
  removeUnreachable(f);
  bool any = false;
  for (int iter = 0; iter < 50; iter++) {
    bool hoisted = false;
    for (const Loop& L : findLoops(f))
      if (hoistFromLoop(f, L)) { hoisted = true; break; }  // CFG changed: recompute loops
    if (!hoisted) break;
    any = true;
  }
  return any;
}

// ======================================================================
// Pass 5: FMA contraction   t = a*b; d = t + c   ->   d = fma(a, b, c)
//         (for ints this is mad.lo.s32, which gives exactly the same bits)
// ======================================================================
bool fmaContraction(Function& f) {
  std::vector<int> uses(f.vregs.size(), 0);
  for (auto& blk : f.blocks)
    for (auto& in : blk.instrs)
      for (auto& o : in.src)
        if (o.isReg()) uses[o.reg]++;
  bool changed = false;
  for (auto& blk : f.blocks) {
    auto& ins = blk.instrs;
    std::vector<char> dead(ins.size(), 0);
    for (size_t i = 0; i < ins.size(); i++) {
      Instr& add = ins[i];
      // float: a*b+c -> fma.rn.f32 ; int: a*b+c -> mad.lo.s32 (bit-identical, one instruction)
      if (add.op != Opcode::Add || (add.type != IRType::F32 && add.type != IRType::I32)) continue;
      for (int k = 0; k < 2; k++) {
        const Operand& m = add.src[k];
        if (!m.isReg() || uses[m.reg] != 1) continue;
        // find the latest definition of m before the add, in this block
        int j = (int)i - 1;
        while (j >= 0 && ins[j].dst != m.reg) j--;
        if (j < 0 || dead[j] || ins[j].op != Opcode::Mul || ins[j].type != add.type) continue;
        // the multiply's inputs must not change between the mul and the add
        bool clobbered = false;
        for (size_t x = j + 1; x < i; x++)
          for (const Operand& o : ins[j].src)
            if (o.isReg() && ins[x].dst == o.reg) clobbered = true;
        if (clobbered) continue;
        Operand c = add.src[1 - k];
        add.op = Opcode::Fma;
        add.src = {ins[j].src[0], ins[j].src[1], c};
        dead[j] = 1;
        changed = true;
        break;
      }
    }
    std::vector<Instr> kept;
    for (size_t i = 0; i < ins.size(); i++)
      if (!dead[i]) kept.push_back(std::move(ins[i]));
    ins = std::move(kept);
  }
  return changed;
}

// ======================================================================
// Pipeline
// ======================================================================
std::vector<PassStat> optimize(Function& f, const OptOptions& opt) {
  std::vector<PassStat> stats;
  auto rec = [&](const char* name) { stats.push_back({name, f.instrCount()}); };
  rec("irgen");
  if (opt.level <= 0) return stats;

  auto cleanup = [&]() {
    for (int round = 0; round < 10; round++) {
      bool c = false;
      c |= simplifyCFG(f);
      c |= localValueNumbering(f);
      c |= globalCopyPropagation(f);
      c |= deadCodeElimination(f);
      c |= coalesceCopies(f);
      if (!c) break;
    }
  };
  simplifyCFG(f);              rec("simplify-cfg");
  localValueNumbering(f);      rec("lvn (fold/propagate/cse)");
  deadCodeElimination(f);      rec("dce");
  globalCopyPropagation(f);
  deadCodeElimination(f);      rec("global copy propagation + dce");
  cleanup();                   rec("cleanup (repeat until stable)");
  if (opt.level >= 2) {
    if (loopInvariantCodeMotion(f)) { rec("licm"); cleanup(); rec("cleanup"); }
  }
  if (opt.fmad && fmaContraction(f)) { deadCodeElimination(f); rec("fma-contraction"); }
  if (coalesceCopies(f)) rec("copy coalescing");
  simplifyCFG(f);
  return stats;
}
