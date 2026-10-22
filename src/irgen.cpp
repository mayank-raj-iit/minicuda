// irgen.cpp - walks the AST and emits IR instructions into basic blocks.
//
// Key ideas:
//  * every source variable gets one virtual register (its "home")
//  * every expression result goes into a fresh virtual register
//  * if/while/for become basic blocks joined by br/condbr
//  * && and || "short-circuit": the right side runs only when needed
#include "irgen.hpp"

#include <unordered_map>

namespace {

IRType irType(const Type& t) {
  if (t.isPtr) return IRType::I64;
  switch (t.base) {
    case BaseType::Int: return IRType::I32;
    case BaseType::Float: return IRType::F32;
    case BaseType::Bool: return IRType::Pred;
    default: return IRType::None;
  }
}

class IRGen {
 public:
  explicit IRGen(const Kernel& k) : k_(k) {}

  Function run() {
    f_.name = k_.name;
    setBlock(newBlock("entry"));
    curLine_ = k_.line;
    // Load every parameter into a register. Pointers are also converted to
    // global-space addresses once, up front (exactly what nvcc does).
    for (const Param& p : k_.params) {
      ParamInfo pi;
      pi.name = p.name;
      pi.type = irType(p.type);
      pi.isPtr = p.type.isPtr;
      pi.elem = p.type.isPtr ? irType(mkType(p.type.base)) : IRType::None;
      f_.params.push_back(pi);
      int r = emitValue(Opcode::LdParam, pi.type, {}, p.sym->paramIndex, p.name);
      if (p.type.isPtr) r = emitValue(Opcode::CvtaGlobal, IRType::I64, {reg(r)}, 0, p.name + ".global");
      home_[p.sym] = r;
    }
    stmt(*k_.body);
    if (!terminated()) emitTerm(Opcode::Ret);
    // blocks that were created but never reached (e.g. after 'break') still
    // need a terminator so the CFG is well formed
    for (size_t b = 0; b < f_.blocks.size(); b++) {
      setBlock((int)b);
      if (!terminated()) emitTerm(Opcode::Ret);
    }
    finalizeLayout();
    return std::move(f_);
  }

 private:
  const Kernel& k_;
  Function f_;
  int cur_ = -1;
  int curLine_ = 0, curCol_ = 0;
  std::unordered_map<const Symbol*, int> home_;       // variable -> its register
  std::unordered_map<const Symbol*, int> sharedIdx_;  // shared array -> index in f_.shared
  struct Loop { int continueTo, breakTo; };
  std::vector<Loop> loops_;
  std::vector<bool> started_;
  std::vector<int> order_;

  // ---- block / instruction helpers ----
  int newBlock(const std::string& name) {
    f_.blocks.push_back(BasicBlock{name, {}});
    return (int)f_.blocks.size() - 1;
  }
  void setBlock(int b) {
    cur_ = b;
    if (b >= (int)started_.size()) started_.resize(b + 1, false);
    if (!started_[b]) { started_[b] = true; order_.push_back(b); }
  }

  // Lay blocks out in the order we started filling them (= source order),
  // so the PTX reads top to bottom like the CUDA source.
  void finalizeLayout() {
    std::vector<int> newId(f_.blocks.size(), -1);
    std::vector<BasicBlock> laid;
    for (int b : order_) { newId[b] = (int)laid.size(); laid.push_back(std::move(f_.blocks[b])); }
    for (auto& blk : laid)
      for (auto& in : blk.instrs) {
        if (in.op == Opcode::Br || in.op == Opcode::CondBr) in.target = newId[in.target];
        if (in.op == Opcode::CondBr) in.target2 = newId[in.target2];
      }
    f_.blocks = std::move(laid);
  }
  bool terminated() const {
    const auto& ins = f_.blocks[cur_].instrs;
    return !ins.empty() && isTerminator(ins.back().op);
  }
  Operand reg(int r) const { return Operand::R(r, f_.vregs[r].type); }

  void append(Instr in) {
    if (terminated()) setBlock(newBlock("unreachable"));  // code after return/break
    in.line = curLine_;
    in.col = curCol_;
    f_.blocks[cur_].instrs.push_back(std::move(in));
  }

  // Emit an instruction that produces a value; returns the new register.
  int emitValue(Opcode op, IRType ty, std::vector<Operand> src, int aux = 0, const std::string& hint = "") {
    Instr in;
    in.op = op;
    in.type = ty;
    in.src = std::move(src);
    in.aux = aux;
    in.dst = f_.newReg(resultType(in), hint);
    int d = in.dst;
    append(std::move(in));
    return d;
  }
  void emitVoid(Opcode op, IRType ty, std::vector<Operand> src) {
    Instr in;
    in.op = op;
    in.type = ty;
    in.src = std::move(src);
    append(std::move(in));
  }
  void emitMov(int dst, Operand v) {
    Instr in;
    in.op = Opcode::Mov;
    in.type = f_.vregs[dst].type;
    in.dst = dst;
    in.src = {v};
    append(std::move(in));
  }
  void emitTerm(Opcode op, int t = -1, int t2 = -1, Operand p = Operand()) {
    Instr in;
    in.op = op;
    in.target = t;
    in.target2 = t2;
    if (op == Opcode::CondBr) in.src = {p};
    append(std::move(in));
  }
  void br(int target) { if (!terminated()) emitTerm(Opcode::Br, target); }

  // ---- statements ----
  void stmt(const Stmt& s) {
    curLine_ = s.line;
    curCol_ = s.col;
    switch (s.kind) {
      case StmtKind::Block:
        for (auto& c : s.stmts) stmt(*c);
        break;

      case StmtKind::VarDecl: {
        int r = f_.newReg(irType(s.declType), s.name);
        home_[s.sym] = r;
        // Uninitialised variables start at 0 so results are deterministic.
        emitMov(r, s.expr ? expr(*s.expr) : (s.declType.isFloat() ? Operand::F(0) : Operand::I(0)));
        break;
      }

      case StmtKind::SharedDecl:
        sharedIdx_[s.sym] = (int)f_.shared.size();
        f_.shared.push_back({k_.name + "_" + s.name, irType(s.declType), s.arraySize});
        break;

      case StmtKind::Assign: assign(s); break;

      case StmtKind::ExprStmt:
        if (s.expr->name == "__syncthreads") emitVoid(Opcode::Bar, IRType::None, {});
        else expr(*s.expr);
        break;

      case StmtKind::If: {
        int thenB = newBlock("if.then");
        int endB = newBlock("if.end");
        int elseB = s.elseS ? newBlock("if.else") : endB;
        cond(*s.expr, thenB, elseB);
        setBlock(thenB);
        stmt(*s.thenS);
        br(endB);
        if (s.elseS) {
          setBlock(elseB);
          stmt(*s.elseS);
          br(endB);
        }
        setBlock(endB);
        break;
      }

      case StmtKind::While: {
        int condB = newBlock("while.cond");
        int bodyB = newBlock("while.body");
        int endB = newBlock("while.end");
        br(condB);
        setBlock(condB);
        curLine_ = s.line;
        cond(*s.expr, bodyB, endB);
        setBlock(bodyB);
        loops_.push_back({condB, endB});
        stmt(*s.body);
        loops_.pop_back();
        br(condB);
        setBlock(endB);
        break;
      }

      case StmtKind::For: {
        if (s.init) stmt(*s.init);
        int condB = newBlock("for.cond");
        int bodyB = newBlock("for.body");
        int stepB = newBlock("for.step");
        int endB = newBlock("for.end");
        br(condB);
        setBlock(condB);
        curLine_ = s.line;
        if (s.expr) cond(*s.expr, bodyB, endB);
        else br(bodyB);
        setBlock(bodyB);
        loops_.push_back({stepB, endB});
        stmt(*s.body);
        loops_.pop_back();
        br(stepB);
        setBlock(stepB);
        if (s.step) stmt(*s.step);
        br(condB);
        setBlock(endB);
        break;
      }

      case StmtKind::Return: emitTerm(Opcode::Ret); break;
      case StmtKind::Break: br(loops_.back().breakTo); emitDeadBlock(); break;
      case StmtKind::Continue: br(loops_.back().continueTo); emitDeadBlock(); break;
    }
  }

  void emitDeadBlock() { setBlock(newBlock("unreachable")); }

  void assign(const Stmt& s) {
    const Expr& t = *s.target;
    auto arith = [&](Operand cur, IRType targetTy) -> Operand {
      // compound assignment: target = (T)((OpT)target op value)
      IRType opTy = irType(s.opType);
      Operand lhs = convertTo(cur, targetTy, opTy);
      Opcode op = s.op == "+=" ? Opcode::Add : s.op == "-=" ? Opcode::Sub : s.op == "*=" ? Opcode::Mul : Opcode::Div;
      Operand rhs = expr(*s.expr);
      Operand res = reg(emitValue(op, opTy, {lhs, rhs}));
      return convertTo(res, opTy, targetTy);
    };
    if (t.kind == ExprKind::Var) {
      int h = home_.at(t.sym);
      Operand v = s.op == "=" ? expr(*s.expr) : arith(reg(h), f_.vregs[h].type);
      emitMov(h, v);
    } else {  // array element
      Addr a = address(t);
      Operand v;
      if (s.op == "=") v = expr(*s.expr);
      else v = arith(reg(emitValue(a.shared ? Opcode::LdShared : Opcode::LdGlobal, a.elem, {a.addr})), a.elem);
      emitVoid(a.shared ? Opcode::StShared : Opcode::StGlobal, a.elem, {a.addr, v});
    }
  }

  // Convert a value between IR types (used by compound assignment).
  Operand convertTo(Operand v, IRType from, IRType to) {
    if (from == to) return v;
    if (from == IRType::I32 && to == IRType::F32) return reg(emitValue(Opcode::CvtI2F, IRType::F32, {v}));
    if (from == IRType::F32 && to == IRType::I32) return reg(emitValue(Opcode::CvtF2I, IRType::I32, {v}));
    return v;
  }

  // ---- addresses ----
  struct Addr { bool shared; Operand addr; IRType elem; };

  // address of a[i] = base + i * sizeof(element)
  Addr address(const Expr& idx) {
    const Symbol* sym = idx.sym;
    Operand i = expr(*idx.kids[1]);
    if (sym->kind == SymKind::Shared) {
      int si = sharedIdx_.at(sym);
      IRType elem = f_.shared[si].elem;
      Operand base = reg(emitValue(Opcode::SharedAddr, IRType::I32, {}, si));
      Operand off = reg(emitValue(Opcode::Mul, IRType::I32, {i, Operand::I(typeSize(elem))}));
      return {true, reg(emitValue(Opcode::Add, IRType::I32, {base, off})), elem};
    }
    IRType elem = irType(mkType(sym->type.base));
    Operand off = reg(emitValue(Opcode::MulWide, IRType::I32, {i, Operand::I(typeSize(elem))}));
    Operand base = reg(home_.at(sym));
    return {false, reg(emitValue(Opcode::Add, IRType::I64, {base, off})), elem};
  }

  // ---- expressions ----
  Operand expr(const Expr& e) {
    switch (e.kind) {
      case ExprKind::IntLit: return Operand::I(e.ival);
      case ExprKind::FloatLit: return Operand::F(e.fval);
      case ExprKind::Var: return reg(home_.at(e.sym));
      case ExprKind::Builtin: {
        int group = e.builtin == Builtin::ThreadIdx ? 0 : e.builtin == Builtin::BlockDim ? 1
                  : e.builtin == Builtin::BlockIdx ? 2 : 3;
        return reg(emitValue(Opcode::SReg, IRType::I32, {}, group * 3 + e.dim));
      }
      case ExprKind::Index: {
        Addr a = address(e);
        return reg(emitValue(a.shared ? Opcode::LdShared : Opcode::LdGlobal, a.elem, {a.addr}));
      }
      case ExprKind::Unary: {
        if (e.op == "+") return expr(*e.kids[0]);
        Operand v = expr(*e.kids[0]);
        if (e.op == "!") return reg(emitValue(Opcode::Not, IRType::Pred, {v}));
        return reg(emitValue(Opcode::Neg, irType(e.type), {v}));
      }
      case ExprKind::Binary: return binary(e);
      case ExprKind::Cast: return cast(e);
      case ExprKind::Call: return call(e);
      case ExprKind::AddrOf: break;  // handled inside atomicAdd
    }
    return Operand::I(0);
  }

  Operand binary(const Expr& e) {
    const std::string& op = e.op;
    if (op == "&&" || op == "||") {
      // value context, e.g.  int ok = a < n && b < m;  ->  materialise 0/1 via branches
      int tmp = f_.newReg(IRType::I32, "logic");
      int tB = newBlock("logic.true"), fB = newBlock("logic.false"), endB = newBlock("logic.end");
      cond(e, tB, fB);
      setBlock(tB); emitMov(tmp, Operand::I(1)); br(endB);
      setBlock(fB); emitMov(tmp, Operand::I(0)); br(endB);
      setBlock(endB);
      return reg(emitValue(Opcode::CmpNE, IRType::I32, {reg(tmp), Operand::I(0)}));
    }
    Operand a = expr(*e.kids[0]);
    Operand b = expr(*e.kids[1]);
    IRType ty = irType(e.kids[0]->type);  // operands already have the same type
    static const std::unordered_map<std::string, Opcode> kOps = {
        {"+", Opcode::Add}, {"-", Opcode::Sub}, {"*", Opcode::Mul}, {"/", Opcode::Div},
        {"%", Opcode::Rem}, {"&", Opcode::And}, {"|", Opcode::Or}, {"^", Opcode::Xor},
        {"<<", Opcode::Shl}, {">>", Opcode::Shr}, {"<", Opcode::CmpLT}, {"<=", Opcode::CmpLE},
        {">", Opcode::CmpGT}, {">=", Opcode::CmpGE}, {"==", Opcode::CmpEQ}, {"!=", Opcode::CmpNE}};
    return reg(emitValue(kOps.at(op), ty, {a, b}));
  }

  Operand cast(const Expr& e) {
    const Expr& in = *e.kids[0];
    IRType from = irType(in.type), to = irType(e.castTo);
    Operand v = expr(in);
    if (from == to) return v;
    if (to == IRType::Pred) {  // x != 0
      Operand zero = from == IRType::F32 ? Operand::F(0) : Operand::I(0);
      return reg(emitValue(Opcode::CmpNE, from, {v, zero}));
    }
    if (from == IRType::Pred) {  // b ? 1 : 0
      if (to == IRType::F32) return reg(emitValue(Opcode::Select, IRType::F32, {v, Operand::F(1), Operand::F(0)}));
      return reg(emitValue(Opcode::Select, IRType::I32, {v, Operand::I(1), Operand::I(0)}));
    }
    return convertTo(v, from, to);
  }

  Operand call(const Expr& e) {
    const std::string& n = e.name;
    if (n == "atomicAdd") {
      Addr a = address(*e.kids[0]->kids[0]);
      Operand v = expr(*e.kids[1]);
      return reg(emitValue(a.shared ? Opcode::AtomAddShared : Opcode::AtomAddGlobal, a.elem, {a.addr, v}));
    }
    std::vector<Operand> args;
    for (auto& k : e.kids) args.push_back(expr(*k));
    IRType ty = irType(e.type);
    if (n == "sqrtf") return reg(emitValue(Opcode::Sqrt, ty, args));
    if (n == "fabsf" || n == "abs") return reg(emitValue(Opcode::Abs, ty, args));
    if (n == "fminf" || n == "min") return reg(emitValue(Opcode::Min, ty, args));
    if (n == "fmaxf" || n == "max") return reg(emitValue(Opcode::Max, ty, args));
    return Operand::I(0);
  }

  // ---- conditions: jump to tB if e is true, else to fB ----
  void cond(const Expr& e, int tB, int fB) {
    if (e.kind == ExprKind::Binary && e.op == "&&") {
      int mid = newBlock("and.rhs");
      cond(*e.kids[0], mid, fB);   // left false -> whole thing false, skip right side
      setBlock(mid);
      cond(*e.kids[1], tB, fB);
      return;
    }
    if (e.kind == ExprKind::Binary && e.op == "||") {
      int mid = newBlock("or.rhs");
      cond(*e.kids[0], tB, mid);   // left true -> whole thing true, skip right side
      setBlock(mid);
      cond(*e.kids[1], tB, fB);
      return;
    }
    if (e.kind == ExprKind::Unary && e.op == "!") {
      cond(*e.kids[0], fB, tB);
      return;
    }
    Operand p = expr(e);
    emitTerm(Opcode::CondBr, tB, fB, p);
  }
};

}  // namespace

Function generateIR(const Kernel& k) { return IRGen(k).run(); }
