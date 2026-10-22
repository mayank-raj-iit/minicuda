// diag.hpp - error reporting shared by every compiler phase.
#pragma once
#include <stdexcept>
#include <string>

// A compile error that remembers where in the source it happened.
// The lexer, parser and semantic checker throw it; main() catches it and
// prints "file:line:col: error: message" (the same style GCC/Clang use).
struct CompileError : std::runtime_error {
  int line, col;
  CompileError(int l, int c, const std::string& msg)
      : std::runtime_error(msg), line(l), col(c) {}
};
