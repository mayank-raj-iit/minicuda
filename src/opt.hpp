// opt.hpp - Phase 5: optimization passes over the IR.
#pragma once
#include <string>
#include <vector>

#include "ir.hpp"

struct OptOptions {
  int level = 2;       // -O0 / -O1 / -O2
  bool fmad = true;    // contract a*b+c into fma (nvcc's default --fmad=true)
};

struct PassStat {
  std::string pass;
  size_t instrs;       // IR instruction count after the pass
};

// Runs the pass pipeline for the chosen level; returns per-pass statistics.
std::vector<PassStat> optimize(Function& f, const OptOptions& opt);

// The individual passes. Each returns true if it changed the function.
bool simplifyCFG(Function& f);              // fold constant branches, drop dead blocks, merge blocks
bool localValueNumbering(Function& f);      // constant folding + propagation, copy propagation, CSE
bool deadCodeElimination(Function& f);      // delete instructions whose result is never used
bool globalCopyPropagation(Function& f);    // x = y (both assigned once): use y everywhere instead of x
bool coalesceCopies(Function& f);           // t = a + b; x = t   ->   x = a + b
bool loopInvariantCodeMotion(Function& f);  // move loop-invariant work into the loop preheader
bool fmaContraction(Function& f);           // mul + add  ->  fma

// Analyses (also used by the PTX emitter and the tests).
struct Liveness {
  std::vector<std::vector<char>> in, out;   // [block][vreg] = live?
};
Liveness computeLiveness(const Function& f);
std::vector<std::vector<int>> computePreds(const Function& f);
