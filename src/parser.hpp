// parser.hpp - Phase 2: turn the token list into an AST (recursive descent).
#pragma once
#include <vector>

#include "ast.hpp"
#include "lexer.hpp"

// Throws CompileError on a syntax error.
Program parseProgram(const std::vector<Token>& tokens);
