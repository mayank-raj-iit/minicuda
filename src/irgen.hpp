// irgen.hpp - Phase 4: lower the checked AST into IR (one Function per kernel).
#pragma once
#include "ast.hpp"
#include "ir.hpp"

Function generateIR(const Kernel& k);
