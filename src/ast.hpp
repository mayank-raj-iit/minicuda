// ast.hpp - the Abstract Syntax Tree: the program as a tree of objects.
#pragma once
#include <memory>
#include <ostream>
#include <string>
#include <vector>

// ---------- types ----------
enum class BaseType { Void, Bool, Int, Float };

struct Type {
  BaseType base = BaseType::Void;
  bool isPtr = false;  // int* / float* (only kernel parameters can be pointers)

  bool isInt() const { return base == BaseType::Int && !isPtr; }
  bool isFloat() const { return base == BaseType::Float && !isPtr; }
  bool isBool() const { return base == BaseType::Bool && !isPtr; }
  bool isScalar() const { return !isPtr && base != BaseType::Void; }
  bool operator==(const Type& o) const { return base == o.base && isPtr == o.isPtr; }
  bool operator!=(const Type& o) const { return !(*this == o); }
  std::string str() const;
};
inline Type mkType(BaseType b, bool ptr = false) { Type t; t.base = b; t.isPtr = ptr; return t; }

// ---------- symbols (named things: parameters, variables, shared arrays) ----------
enum class SymKind { Param, Local, Shared };

struct Symbol {
  std::string name;
  SymKind kind = SymKind::Local;
  Type type;                // for Shared: the element type
  long long arraySize = 0;  // for Shared
  int paramIndex = -1;      // for Param
  int line = 0, col = 0;
};

// ---------- expressions ----------
enum class Builtin { ThreadIdx, BlockIdx, BlockDim, GridDim };

enum class ExprKind {
  IntLit,    // 42
  FloatLit,  // 2.5f
  Var,       // x
  Index,     // a[i]        kids = {base Var, index}
  Unary,     // -x  !x      kids = {operand}
  Binary,    // a + b       kids = {lhs, rhs}
  Cast,      // (float)x    kids = {operand}  (sema also inserts implicit casts)
  Call,      // sqrtf(x)    kids = args
  Builtin,   // threadIdx.x
  AddrOf,    // &a[i]       kids = {Index}   (only inside atomicAdd)
};

struct Expr {
  ExprKind kind;
  int line = 0, col = 0;
  long long ival = 0;   // IntLit
  float fval = 0;       // FloatLit
  std::string name;     // Var / Call
  std::string op;       // Unary / Binary operator text, e.g. "+", "&&"
  Type castTo;          // Cast
  Builtin builtin = Builtin::ThreadIdx;
  int dim = 0;          // Builtin: 0 = .x, 1 = .y, 2 = .z
  std::vector<std::unique_ptr<Expr>> kids;

  // Filled in by semantic analysis (sema.cpp):
  Type type;
  Symbol* sym = nullptr;  // Var and Index: the variable / array being used
};
using ExprPtr = std::unique_ptr<Expr>;

// ---------- statements ----------
enum class StmtKind { Block, VarDecl, SharedDecl, Assign, ExprStmt, If, For, While, Return, Break, Continue };

struct Stmt {
  StmtKind kind;
  int line = 0, col = 0;
  bool newScope = true;                     // Block: false for "int a = 1, b = 2;" groups
  std::vector<std::unique_ptr<Stmt>> stmts; // Block
  Type declType;                            // VarDecl / SharedDecl
  std::string name;                         // VarDecl / SharedDecl
  long long arraySize = 0;                  // SharedDecl
  ExprPtr target;                           // Assign: the left-hand side
  std::string op;                           // Assign: "=", "+=", "-=", "*=", "/="
  ExprPtr expr;                             // VarDecl init, Assign value, ExprStmt, If/While/For condition
  std::unique_ptr<Stmt> thenS, elseS;       // If
  std::unique_ptr<Stmt> init, step, body;   // For (init/step may be null); While uses body

  // Filled in by sema:
  Symbol* sym = nullptr;  // VarDecl / SharedDecl
  Type opType;            // Assign with "+=" etc: the type the arithmetic is done in
};
using StmtPtr = std::unique_ptr<Stmt>;

// ---------- kernels and the whole program ----------
struct Param {
  Type type;
  std::string name;
  int line = 0, col = 0;
  Symbol* sym = nullptr;
};

struct Kernel {
  std::string name;
  int line = 0, col = 0;
  std::vector<Param> params;
  StmtPtr body;
  std::vector<std::unique_ptr<Symbol>> symbols;  // owns every Symbol of this kernel
};

struct Program {
  std::vector<std::unique_ptr<Kernel>> kernels;
};

void dumpAst(const Program& p, std::ostream& os);
