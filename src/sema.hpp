// sema.hpp - Phase 3: semantic analysis (names, scopes, types).
#pragma once
#include "ast.hpp"

// Checks the program and annotates it:
//  * every Var/Index gets its Symbol, every Expr gets its Type
//  * implicit conversions (int -> float, value -> bool) become explicit Cast nodes
// Throws CompileError for undeclared names, type errors, misuse of builtins, etc.
void checkProgram(Program& prog);
