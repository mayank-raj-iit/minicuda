// ptxgen.hpp - Phase 6: emit NVIDIA PTX assembly from the IR.
#pragma once
#include <string>
#include <vector>

#include "ir.hpp"

struct PtxOptions {
  std::string arch = "sm_75";   // T4 GPU on Google Colab = sm_75
  bool lineInfo = false;        // emit .file/.loc so tools map SASS back to source lines
  std::string sourceName = "input.cu";
};

// Rewrites operands PTX cannot encode (e.g. an immediate as the first source
// of sub/setp, or as the value of a store) into registers.
void legalizeForPtx(Function& f);

// Produces a complete .ptx module containing every kernel.
std::string emitPtx(std::vector<Function>& fns, const PtxOptions& opt, size_t* instrCount = nullptr);

std::string ptxVersionFor(const std::string& arch);
