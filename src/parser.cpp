// parser.cpp - a hand-written recursive-descent parser.
//
// Grammar (simplified):
//   program   := kernel*
//   kernel    := ['extern' "C"] '__global__' 'void' IDENT '(' params ')' block
//   param     := type IDENT
//   type      := ['const'] ('int' | 'float') ['*' ['const'] ['__restrict__']]
//   block     := '{' stmt* '}'
//   stmt      := block | if | for | while | 'return' ';' | 'break' ';' | 'continue' ';'
//              | '__shared__' type IDENT '[' INT ']' ';'
//              | type IDENT ['=' expr] (',' IDENT ['=' expr])* ';'
//              | simple ';'
//   simple    := lvalue ('=' | '+=' | '-=' | '*=' | '/=') expr | lvalue '++' | '++' lvalue | call
//   expr      := binary expression with C precedence, then unary, postfix, primary
#include "parser.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <sstream>

#include "diag.hpp"

namespace {

class Parser {
 public:
  explicit Parser(const std::vector<Token>& t) : t_(t) {}

  Program program() {
    Program prog;
    while (!check(Tok::End)) prog.kernels.push_back(kernel());
    return prog;
  }

 private:
  const std::vector<Token>& t_;
  size_t p_ = 0;

  // ---- small helpers ----
  const Token& cur() const { return t_[p_]; }
  const Token& ahead(size_t k) const { return t_[std::min(p_ + k, t_.size() - 1)]; }
  bool check(Tok k) const { return cur().kind == k; }
  bool accept(Tok k) {
    if (!check(k)) return false;
    p_++;
    return true;
  }
  const Token& expect(Tok k, const std::string& what) {
    if (!check(k)) error("expected " + what + " but found '" + cur().text + "'");
    return t_[p_++];
  }
  [[noreturn]] void error(const std::string& msg) const {
    throw CompileError(cur().line, cur().col, msg);
  }
  bool isWord(const Token& t, const char* w) const { return t.kind == Tok::Ident && t.text == w; }

  // Does a type start here?  int / float / const int ...
  bool typeStartsAt(size_t k) const {
    const Token& a = ahead(k);
    if (a.kind == Tok::KwInt || a.kind == Tok::KwFloat) return true;
    if (isWord(a, "const")) {
      Tok n = ahead(k + 1).kind;
      return n == Tok::KwInt || n == Tok::KwFloat;
    }
    return false;
  }

  template <class T>
  std::unique_ptr<T> node(int line, int col) {
    auto n = std::make_unique<T>();
    n->line = line;
    n->col = col;
    return n;
  }
  ExprPtr expr(ExprKind k, const Token& at) {
    auto e = node<Expr>(at.line, at.col);
    e->kind = k;
    return e;
  }
  StmtPtr stmt(StmtKind k, const Token& at) {
    auto s = node<Stmt>(at.line, at.col);
    s->kind = k;
    return s;
  }

  // ---- declarations ----
  Type type() {
    if (isWord(cur(), "const")) p_++;
    Type ty;
    if (accept(Tok::KwInt)) ty.base = BaseType::Int;
    else if (accept(Tok::KwFloat)) ty.base = BaseType::Float;
    else error("expected a type ('int' or 'float') but found '" + cur().text + "'");
    if (accept(Tok::Star)) {
      ty.isPtr = true;
      while (isWord(cur(), "const") || isWord(cur(), "__restrict__")) p_++;
    }
    return ty;
  }

  std::unique_ptr<Kernel> kernel() {
    if (accept(Tok::KwExtern)) {
      const Token& s = expect(Tok::StrLit, "\"C\" after 'extern'");
      if (s.text != "\"C\"") error("only extern \"C\" is supported");
    }
    if (check(Tok::KwVoid) || typeStartsAt(0))
      error("only __global__ kernels are supported (device functions are not)");
    expect(Tok::KwGlobal, "'__global__'");
    expect(Tok::KwVoid, "'void' (a kernel must return void)");
    const Token& name = expect(Tok::Ident, "kernel name");
    auto k = std::make_unique<Kernel>();
    k->name = name.text;
    k->line = name.line;
    k->col = name.col;
    expect(Tok::LParen, "'('");
    if (!check(Tok::RParen)) {
      do {
        Param prm;
        prm.line = cur().line;
        prm.col = cur().col;
        prm.type = type();
        prm.name = expect(Tok::Ident, "parameter name").text;
        k->params.push_back(prm);
      } while (accept(Tok::Comma));
    }
    expect(Tok::RParen, "')'");
    if (!check(Tok::LBrace)) error("expected '{' to start the kernel body");
    k->body = block();
    return k;
  }

  // ---- statements ----
  StmtPtr block() {
    const Token& open = expect(Tok::LBrace, "'{'");
    auto b = stmt(StmtKind::Block, open);
    while (!check(Tok::RBrace)) {
      if (check(Tok::End)) error("missing '}' (block opened at line " + std::to_string(open.line) + ")");
      b->stmts.push_back(statement());
    }
    p_++;  // '}'
    return b;
  }

  StmtPtr statement() {
    const Token& at = cur();
    switch (at.kind) {
      case Tok::LBrace: return block();
      case Tok::KwIf: return ifStmt();
      case Tok::KwFor: return forStmt();
      case Tok::KwWhile: return whileStmt();
      case Tok::KwReturn: {
        p_++;
        if (!check(Tok::Semi)) error("kernels return void: use 'return;'");
        p_++;
        return stmt(StmtKind::Return, at);
      }
      case Tok::KwBreak: p_++; expect(Tok::Semi, "';'"); return stmt(StmtKind::Break, at);
      case Tok::KwContinue: p_++; expect(Tok::Semi, "';'"); return stmt(StmtKind::Continue, at);
      case Tok::KwShared: return sharedDecl();
      case Tok::Semi: p_++; return stmt(StmtKind::Block, at);  // empty statement
      default: break;
    }
    if (typeStartsAt(0)) {
      auto s = varDecl();
      expect(Tok::Semi, "';' after declaration");
      return s;
    }
    auto s = simpleStmt();
    expect(Tok::Semi, "';'");
    return s;
  }

  // int a = 1, b;  -> one VarDecl, or a scope-less Block holding several
  StmtPtr varDecl() {
    const Token& at = cur();
    Type ty = type();
    auto group = stmt(StmtKind::Block, at);
    group->newScope = false;
    do {
      const Token& name = expect(Tok::Ident, "variable name");
      auto d = stmt(StmtKind::VarDecl, name);
      d->declType = ty;
      d->name = name.text;
      if (check(Tok::LBracket)) error("local arrays are not supported (use __shared__ or a pointer parameter)");
      if (accept(Tok::Assign)) d->expr = expression();
      group->stmts.push_back(std::move(d));
    } while (accept(Tok::Comma));
    if (group->stmts.size() == 1) return std::move(group->stmts[0]);
    return group;
  }

  StmtPtr sharedDecl() {
    const Token& at = expect(Tok::KwShared, "'__shared__'");
    auto s = stmt(StmtKind::SharedDecl, at);
    s->declType = type();
    if (s->declType.isPtr) error("__shared__ pointers are not supported");
    s->name = expect(Tok::Ident, "shared array name").text;
    expect(Tok::LBracket, "'[' (a __shared__ variable must be an array)");
    const Token& n = expect(Tok::IntLit, "array size (an integer constant)");
    s->arraySize = std::strtoll(n.text.c_str(), nullptr, 10);
    expect(Tok::RBracket, "']'");
    expect(Tok::Semi, "';'");
    return s;
  }

  // assignment, x++, ++x, or a call such as __syncthreads()
  StmtPtr simpleStmt() {
    const Token& at = cur();
    if (check(Tok::PlusPlus) || check(Tok::MinusMinus)) {
      bool inc = cur().kind == Tok::PlusPlus;
      p_++;
      return incDec(at, unary(), inc);
    }
    ExprPtr e = expression();
    Tok k = cur().kind;
    if (k == Tok::Assign || k == Tok::PlusAssign || k == Tok::MinusAssign ||
        k == Tok::StarAssign || k == Tok::SlashAssign) {
      auto s = stmt(StmtKind::Assign, at);
      s->op = cur().text;
      p_++;
      s->target = std::move(e);
      s->expr = expression();
      return s;
    }
    if (k == Tok::PlusPlus || k == Tok::MinusMinus) {
      p_++;
      return incDec(at, std::move(e), k == Tok::PlusPlus);
    }
    auto s = stmt(StmtKind::ExprStmt, at);
    s->expr = std::move(e);
    return s;
  }

  StmtPtr incDec(const Token& at, ExprPtr target, bool inc) {
    auto s = stmt(StmtKind::Assign, at);
    s->op = inc ? "+=" : "-=";
    s->target = std::move(target);
    auto one = expr(ExprKind::IntLit, at);
    one->ival = 1;
    s->expr = std::move(one);
    return s;
  }

  StmtPtr ifStmt() {
    const Token& at = expect(Tok::KwIf, "'if'");
    auto s = stmt(StmtKind::If, at);
    expect(Tok::LParen, "'(' after 'if'");
    s->expr = expression();
    expect(Tok::RParen, "')'");
    s->thenS = statement();
    if (accept(Tok::KwElse)) s->elseS = statement();
    return s;
  }

  StmtPtr whileStmt() {
    const Token& at = expect(Tok::KwWhile, "'while'");
    auto s = stmt(StmtKind::While, at);
    expect(Tok::LParen, "'(' after 'while'");
    s->expr = expression();
    expect(Tok::RParen, "')'");
    s->body = statement();
    return s;
  }

  StmtPtr forStmt() {
    const Token& at = expect(Tok::KwFor, "'for'");
    auto s = stmt(StmtKind::For, at);
    expect(Tok::LParen, "'(' after 'for'");
    if (!check(Tok::Semi)) s->init = typeStartsAt(0) ? varDecl() : simpleStmt();
    expect(Tok::Semi, "';' after for-loop initializer");
    if (!check(Tok::Semi)) s->expr = expression();
    expect(Tok::Semi, "';' after for-loop condition");
    if (!check(Tok::RParen)) s->step = simpleStmt();
    expect(Tok::RParen, "')'");
    s->body = statement();
    return s;
  }

  // ---- expressions (precedence climbing) ----
  static int precedence(Tok k) {
    switch (k) {
      case Tok::OrOr: return 1;
      case Tok::AndAnd: return 2;
      case Tok::Pipe: return 3;
      case Tok::Caret: return 4;
      case Tok::Amp: return 5;
      case Tok::EqEq: case Tok::NotEq: return 6;
      case Tok::Less: case Tok::LessEq: case Tok::Greater: case Tok::GreaterEq: return 7;
      case Tok::ShiftL: case Tok::ShiftR: return 8;
      case Tok::Plus: case Tok::Minus: return 9;
      case Tok::Star: case Tok::Slash: case Tok::Percent: return 10;
      default: return -1;
    }
  }

  ExprPtr expression() { return binary(1); }

  ExprPtr binary(int minPrec) {
    ExprPtr lhs = unary();
    while (true) {
      int prec = precedence(cur().kind);
      if (prec < minPrec) return lhs;
      const Token& opTok = t_[p_++];
      ExprPtr rhs = binary(prec + 1);  // +1 makes operators left-associative
      auto b = expr(ExprKind::Binary, opTok);
      b->op = opTok.text;
      b->kids.push_back(std::move(lhs));
      b->kids.push_back(std::move(rhs));
      lhs = std::move(b);
    }
  }

  ExprPtr unary() {
    const Token& at = cur();
    if (check(Tok::Minus) || check(Tok::Bang) || check(Tok::Plus)) {
      p_++;
      auto u = expr(ExprKind::Unary, at);
      u->op = at.text;
      u->kids.push_back(unary());
      return u;
    }
    if (check(Tok::Amp)) {
      p_++;
      auto a = expr(ExprKind::AddrOf, at);
      a->kids.push_back(unary());
      return a;
    }
    if (check(Tok::PlusPlus) || check(Tok::MinusMinus))
      error("'++'/'--' can only be used as a statement, e.g. i++;");
    // cast: (int)x  (float)x
    if (check(Tok::LParen) && typeStartsAt(1)) {
      p_++;
      Type ty = type();
      expect(Tok::RParen, "')' after cast type");
      auto c = expr(ExprKind::Cast, at);
      c->castTo = ty;
      c->kids.push_back(unary());
      return c;
    }
    return postfix();
  }

  ExprPtr postfix() {
    ExprPtr e = primary();
    while (check(Tok::LBracket)) {
      const Token& at = cur();
      p_++;
      auto idx = expr(ExprKind::Index, at);
      idx->kids.push_back(std::move(e));
      idx->kids.push_back(expression());
      expect(Tok::RBracket, "']'");
      e = std::move(idx);
    }
    return e;
  }

  ExprPtr primary() {
    const Token& at = cur();
    switch (at.kind) {
      case Tok::IntLit: {
        p_++;
        auto e = expr(ExprKind::IntLit, at);
        errno = 0;
        long long v = std::strtoll(at.text.c_str(), nullptr, 10);
        if (errno != 0 || v > INT_MAX) throw CompileError(at.line, at.col, "integer literal too large for int");
        e->ival = v;
        return e;
      }
      case Tok::FloatLit: {
        p_++;
        auto e = expr(ExprKind::FloatLit, at);
        e->fval = std::strtof(at.text.c_str(), nullptr);  // strtof stops at the 'f' suffix
        return e;
      }
      case Tok::LParen: {
        p_++;
        ExprPtr e = expression();
        expect(Tok::RParen, "')'");
        return e;
      }
      case Tok::Ident: {
        p_++;
        // threadIdx.x / blockIdx.y / blockDim.z / gridDim.x
        static const struct { const char* n; Builtin b; } kB[] = {
            {"threadIdx", Builtin::ThreadIdx}, {"blockIdx", Builtin::BlockIdx},
            {"blockDim", Builtin::BlockDim},   {"gridDim", Builtin::GridDim}};
        for (const auto& b : kB) {
          if (at.text == b.n) {
            expect(Tok::Dot, std::string("'.' after ") + b.n);
            const Token& m = expect(Tok::Ident, "x, y or z");
            if (m.text != "x" && m.text != "y" && m.text != "z")
              throw CompileError(m.line, m.col, "expected x, y or z after '.'");
            auto e = expr(ExprKind::Builtin, at);
            e->builtin = b.b;
            e->dim = m.text[0] - 'x';
            return e;
          }
        }
        if (accept(Tok::LParen)) {  // function call
          auto c = expr(ExprKind::Call, at);
          c->name = at.text;
          if (!check(Tok::RParen)) {
            do c->kids.push_back(expression());
            while (accept(Tok::Comma));
          }
          expect(Tok::RParen, "')' after call arguments");
          return c;
        }
        auto v = expr(ExprKind::Var, at);
        v->name = at.text;
        return v;
      }
      default:
        error("expected an expression but found '" + at.text + "'");
    }
  }
};

void dumpExpr(const Expr& e, std::ostream& os) {
  switch (e.kind) {
    case ExprKind::IntLit: os << e.ival; break;
    case ExprKind::FloatLit: os << e.fval << "f"; break;
    case ExprKind::Var: os << e.name; break;
    case ExprKind::Index: dumpExpr(*e.kids[0], os); os << "["; dumpExpr(*e.kids[1], os); os << "]"; break;
    case ExprKind::Unary: os << "(" << e.op; dumpExpr(*e.kids[0], os); os << ")"; break;
    case ExprKind::Binary:
      os << "("; dumpExpr(*e.kids[0], os); os << " " << e.op << " "; dumpExpr(*e.kids[1], os); os << ")";
      break;
    case ExprKind::Cast: os << "(" << e.castTo.str() << ")"; dumpExpr(*e.kids[0], os); break;
    case ExprKind::Call:
      os << e.name << "(";
      for (size_t i = 0; i < e.kids.size(); i++) { if (i) os << ", "; dumpExpr(*e.kids[i], os); }
      os << ")";
      break;
    case ExprKind::Builtin: {
      static const char* n[] = {"threadIdx", "blockIdx", "blockDim", "gridDim"};
      os << n[(int)e.builtin] << "." << char('x' + e.dim);
      break;
    }
    case ExprKind::AddrOf: os << "&"; dumpExpr(*e.kids[0], os); break;
  }
}

void dumpStmt(const Stmt& s, std::ostream& os, int depth) {
  std::string pad(depth * 2, ' ');
  switch (s.kind) {
    case StmtKind::Block:
      os << pad << (s.newScope ? "Block" : "DeclGroup") << "\n";
      for (auto& c : s.stmts) dumpStmt(*c, os, depth + 1);
      break;
    case StmtKind::VarDecl:
      os << pad << "VarDecl " << s.declType.str() << " " << s.name;
      if (s.expr) { os << " = "; dumpExpr(*s.expr, os); }
      os << "\n";
      break;
    case StmtKind::SharedDecl:
      os << pad << "SharedDecl " << s.declType.str() << " " << s.name << "[" << s.arraySize << "]\n";
      break;
    case StmtKind::Assign:
      os << pad << "Assign "; dumpExpr(*s.target, os); os << " " << s.op << " "; dumpExpr(*s.expr, os); os << "\n";
      break;
    case StmtKind::ExprStmt: os << pad << "ExprStmt "; dumpExpr(*s.expr, os); os << "\n"; break;
    case StmtKind::If:
      os << pad << "If "; dumpExpr(*s.expr, os); os << "\n";
      dumpStmt(*s.thenS, os, depth + 1);
      if (s.elseS) { os << pad << "Else\n"; dumpStmt(*s.elseS, os, depth + 1); }
      break;
    case StmtKind::For:
      os << pad << "For\n";
      if (s.init) dumpStmt(*s.init, os, depth + 1);
      if (s.expr) { os << pad << "  cond "; dumpExpr(*s.expr, os); os << "\n"; }
      if (s.step) dumpStmt(*s.step, os, depth + 1);
      dumpStmt(*s.body, os, depth + 1);
      break;
    case StmtKind::While:
      os << pad << "While "; dumpExpr(*s.expr, os); os << "\n";
      dumpStmt(*s.body, os, depth + 1);
      break;
    case StmtKind::Return: os << pad << "Return\n"; break;
    case StmtKind::Break: os << pad << "Break\n"; break;
    case StmtKind::Continue: os << pad << "Continue\n"; break;
  }
}

}  // namespace

std::string Type::str() const {
  const char* b = base == BaseType::Int ? "int" : base == BaseType::Float ? "float"
                : base == BaseType::Bool ? "bool" : "void";
  return std::string(b) + (isPtr ? "*" : "");
}

Program parseProgram(const std::vector<Token>& tokens) { return Parser(tokens).program(); }

void dumpAst(const Program& p, std::ostream& os) {
  for (auto& k : p.kernels) {
    os << "Kernel " << k->name << "(";
    for (size_t i = 0; i < k->params.size(); i++) {
      if (i) os << ", ";
      os << k->params[i].type.str() << " " << k->params[i].name;
    }
    os << ")\n";
    dumpStmt(*k->body, os, 1);
  }
}
