// sema.cpp - semantic analysis: scopes, symbol lookup and type checking.
#include "sema.hpp"

#include <set>
#include <unordered_map>

#include "diag.hpp"

namespace {

const Type kInt = mkType(BaseType::Int);
const Type kFloat = mkType(BaseType::Float);
const Type kBool = mkType(BaseType::Bool);
const Type kVoid = mkType(BaseType::Void);

class Sema {
 public:
  void kernel(Kernel& k) {
    k_ = &k;
    scopes_.clear();
    scopes_.emplace_back();
    for (size_t i = 0; i < k.params.size(); i++) {
      Param& p = k.params[i];
      if (p.type.base == BaseType::Void) throw CompileError(p.line, p.col, "parameter cannot be void");
      p.sym = declare(p.name, SymKind::Param, p.type, p.line, p.col);
      p.sym->paramIndex = (int)i;
    }
    stmt(*k.body);
  }

 private:
  Kernel* k_ = nullptr;
  std::vector<std::unordered_map<std::string, Symbol*>> scopes_;
  int loopDepth_ = 0;

  [[noreturn]] static void fail(int line, int col, const std::string& m) { throw CompileError(line, col, m); }

  Symbol* declare(const std::string& name, SymKind kind, Type ty, int line, int col) {
    auto& scope = scopes_.back();
    if (scope.count(name)) fail(line, col, "'" + name + "' is already declared in this scope");
    auto s = std::make_unique<Symbol>();
    s->name = name;
    s->kind = kind;
    s->type = ty;
    s->line = line;
    s->col = col;
    Symbol* raw = s.get();
    k_->symbols.push_back(std::move(s));
    scope[name] = raw;
    return raw;
  }

  Symbol* lookup(const std::string& name) {
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      auto f = it->find(name);
      if (f != it->end()) return f->second;
    }
    return nullptr;
  }

  // Wrap e in a Cast node so that its type becomes `to`.
  static void convert(ExprPtr& e, Type to) {
    if (e->type == to) return;
    if (!e->type.isScalar() || !to.isScalar())
      fail(e->line, e->col, "cannot convert '" + e->type.str() + "' to '" + to.str() + "'");
    auto c = std::make_unique<Expr>();
    c->kind = ExprKind::Cast;
    c->line = e->line;
    c->col = e->col;
    c->castTo = to;
    c->type = to;
    c->kids.push_back(std::move(e));
    e = std::move(c);
  }

  // Usual arithmetic conversion: if either side is float, the result is float.
  static Type arith(Type a, Type b) { return (a.isFloat() || b.isFloat()) ? kFloat : kInt; }

  void requireScalar(const ExprPtr& e, const char* what) {
    if (e->type.isPtr) fail(e->line, e->col, std::string("pointer used where ") + what + " is needed (index it: p[i])");
    if (!e->type.isScalar()) fail(e->line, e->col, std::string("expected ") + what);
  }

  // ---------- statements ----------
  void scoped(Stmt& s) {
    scopes_.emplace_back();
    stmt(s);
    scopes_.pop_back();
  }

  void stmt(Stmt& s) {
    switch (s.kind) {
      case StmtKind::Block:
        if (s.newScope) scopes_.emplace_back();
        for (auto& c : s.stmts) stmt(*c);
        if (s.newScope) scopes_.pop_back();
        break;

      case StmtKind::VarDecl:
        if (s.declType.isPtr) fail(s.line, s.col, "local pointer variables are not supported");
        if (s.expr) {
          expr(s.expr);
          requireScalar(s.expr, "a value");
          convert(s.expr, s.declType);
        }
        s.sym = declare(s.name, SymKind::Local, s.declType, s.line, s.col);
        break;

      case StmtKind::SharedDecl:
        if (s.arraySize <= 0 || s.arraySize > 12288)
          fail(s.line, s.col, "__shared__ array size must be between 1 and 12288 elements");
        s.sym = declare(s.name, SymKind::Shared, s.declType, s.line, s.col);
        s.sym->arraySize = s.arraySize;
        break;

      case StmtKind::Assign: {
        Expr& t = *s.target;
        if (t.kind == ExprKind::Var) {
          Symbol* sym = lookup(t.name);
          if (!sym) fail(t.line, t.col, "use of undeclared variable '" + t.name + "'");
          if (sym->kind == SymKind::Shared) fail(t.line, t.col, "cannot assign to a whole __shared__ array; assign to an element");
          if (sym->type.isPtr) fail(t.line, t.col, "assigning to pointer parameters is not supported");
        } else if (t.kind != ExprKind::Index) {
          fail(t.line, t.col, "left side of assignment must be a variable or an array element");
        }
        expr(s.target);
        expr(s.expr);
        requireScalar(s.expr, "a value");
        if (s.op == "=") {
          convert(s.expr, s.target->type);
        } else {
          s.opType = arith(s.target->type, s.expr->type);
          convert(s.expr, s.opType);
        }
        break;
      }

      case StmtKind::ExprStmt: {
        Expr& e = *s.expr;
        if (e.kind != ExprKind::Call) fail(e.line, e.col, "statement has no effect");
        if (e.name == "__syncthreads") {
          if (!e.kids.empty()) fail(e.line, e.col, "__syncthreads() takes no arguments");
          e.type = kVoid;
        } else {
          expr(s.expr);
        }
        break;
      }

      case StmtKind::If:
        cond(s.expr);
        scoped(*s.thenS);
        if (s.elseS) scoped(*s.elseS);
        break;

      case StmtKind::While:
        cond(s.expr);
        loopDepth_++;
        scoped(*s.body);
        loopDepth_--;
        break;

      case StmtKind::For:
        scopes_.emplace_back();
        if (s.init) stmt(*s.init);
        if (s.expr) cond(s.expr);
        if (s.step) stmt(*s.step);
        loopDepth_++;
        scoped(*s.body);
        loopDepth_--;
        scopes_.pop_back();
        break;

      case StmtKind::Return: break;
      case StmtKind::Break:
      case StmtKind::Continue:
        if (loopDepth_ == 0) fail(s.line, s.col, "'break'/'continue' outside of a loop");
        break;
    }
  }

  void cond(ExprPtr& e) {
    expr(e);
    requireScalar(e, "a condition");
    convert(e, kBool);
  }

  // ---------- expressions ----------
  void expr(ExprPtr& ep) {
    Expr& e = *ep;
    switch (e.kind) {
      case ExprKind::IntLit: e.type = kInt; return;
      case ExprKind::FloatLit: e.type = kFloat; return;
      case ExprKind::Builtin: e.type = kInt; return;

      case ExprKind::Var: {
        Symbol* sym = lookup(e.name);
        if (!sym) fail(e.line, e.col, "use of undeclared variable '" + e.name + "'");
        if (sym->kind == SymKind::Shared) fail(e.line, e.col, "__shared__ array '" + e.name + "' must be indexed, e.g. " + e.name + "[i]");
        e.sym = sym;
        e.type = sym->type;
        return;
      }

      case ExprKind::Index: {
        Expr& base = *e.kids[0];
        if (base.kind != ExprKind::Var) fail(base.line, base.col, "only named arrays can be indexed");
        Symbol* sym = lookup(base.name);
        if (!sym) fail(base.line, base.col, "use of undeclared variable '" + base.name + "'");
        if (sym->kind == SymKind::Shared) e.type = sym->type;
        else if (sym->type.isPtr) e.type = mkType(sym->type.base);
        else fail(base.line, base.col, "'" + base.name + "' is not a pointer or __shared__ array, so it cannot be indexed");
        base.sym = sym;
        base.type = sym->type;
        e.sym = sym;
        expr(e.kids[1]);
        if (e.kids[1]->type.isBool()) convert(e.kids[1], kInt);
        if (!e.kids[1]->type.isInt()) fail(e.kids[1]->line, e.kids[1]->col, "array index must be an int");
        return;
      }

      case ExprKind::Unary: {
        expr(e.kids[0]);
        requireScalar(e.kids[0], "a number");
        if (e.op == "!") {
          convert(e.kids[0], kBool);
          e.type = kBool;
        } else {  // - or +
          if (e.kids[0]->type.isBool()) convert(e.kids[0], kInt);
          e.type = e.kids[0]->type;
        }
        return;
      }

      case ExprKind::Binary: {
        expr(e.kids[0]);
        expr(e.kids[1]);
        requireScalar(e.kids[0], "a number");
        requireScalar(e.kids[1], "a number");
        const std::string& op = e.op;
        if (op == "&&" || op == "||") {
          convert(e.kids[0], kBool);
          convert(e.kids[1], kBool);
          e.type = kBool;
          return;
        }
        for (auto& k : e.kids)
          if (k->type.isBool()) convert(k, kInt);
        if (op == "%" || op == "&" || op == "|" || op == "^" || op == "<<" || op == ">>") {
          if (!e.kids[0]->type.isInt() || !e.kids[1]->type.isInt())
            fail(e.line, e.col, "operator '" + op + "' needs int operands");
          e.type = kInt;
          return;
        }
        Type t = arith(e.kids[0]->type, e.kids[1]->type);
        convert(e.kids[0], t);
        convert(e.kids[1], t);
        bool isCompare = op == "<" || op == "<=" || op == ">" || op == ">=" || op == "==" || op == "!=";
        e.type = isCompare ? kBool : t;
        return;
      }

      case ExprKind::Cast: {
        expr(e.kids[0]);
        requireScalar(e.kids[0], "a number");
        if (e.castTo.isPtr) fail(e.line, e.col, "pointer casts are not supported");
        e.type = e.castTo;
        if (e.kids[0]->type == e.castTo) {  // (int)i is a no-op: drop the cast node
          ExprPtr inner = std::move(e.kids[0]);
          ep = std::move(inner);
        }
        return;
      }

      case ExprKind::AddrOf:
        fail(e.line, e.col, "'&' is only supported as the first argument of atomicAdd");

      case ExprKind::Call: call(ep); return;
    }
  }

  void call(ExprPtr& ep) {
    Expr& e = *ep;
    auto nargs = [&](size_t n) {
      if (e.kids.size() != n)
        fail(e.line, e.col, e.name + "() expects " + std::to_string(n) + " argument(s)");
    };
    if (e.name == "__syncthreads") fail(e.line, e.col, "__syncthreads() must be used as a statement");
    if (e.name == "atomicAdd") {
      nargs(2);
      Expr& a = *e.kids[0];
      if (a.kind != ExprKind::AddrOf || a.kids[0]->kind != ExprKind::Index)
        fail(a.line, a.col, "atomicAdd's first argument must look like &array[index]");
      expr(a.kids[0]);
      a.type = mkType(a.kids[0]->type.base, true);
      expr(e.kids[1]);
      requireScalar(e.kids[1], "a value");
      convert(e.kids[1], a.kids[0]->type);
      e.type = a.kids[0]->type;
      return;
    }
    for (auto& k : e.kids) {
      expr(k);
      requireScalar(k, "a number");
    }
    if (e.name == "sqrtf" || e.name == "fabsf") {
      nargs(1);
      convert(e.kids[0], kFloat);
      e.type = kFloat;
    } else if (e.name == "fminf" || e.name == "fmaxf") {
      nargs(2);
      convert(e.kids[0], kFloat);
      convert(e.kids[1], kFloat);
      e.type = kFloat;
    } else if (e.name == "min" || e.name == "max") {
      nargs(2);
      for (auto& k : e.kids)
        if (k->type.isBool()) convert(k, kInt);
      Type t = arith(e.kids[0]->type, e.kids[1]->type);
      convert(e.kids[0], t);
      convert(e.kids[1], t);
      e.type = t;
    } else if (e.name == "abs") {
      nargs(1);
      if (e.kids[0]->type.isBool()) convert(e.kids[0], kInt);
      if (!e.kids[0]->type.isInt()) fail(e.line, e.col, "abs() takes an int; use fabsf() for float");
      e.type = kInt;
    } else {
      fail(e.line, e.col, "unknown function '" + e.name +
                              "' (supported: atomicAdd, sqrtf, fabsf, fminf, fmaxf, min, max, abs, __syncthreads)");
    }
  }
};

}  // namespace

void checkProgram(Program& prog) {
  std::set<std::string> names;
  for (auto& k : prog.kernels) {
    if (!names.insert(k->name).second) throw CompileError(k->line, k->col, "kernel '" + k->name + "' is defined twice");
    Sema().kernel(*k);
  }
}
