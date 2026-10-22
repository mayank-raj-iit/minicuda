#!/usr/bin/env python3
"""MiniCUDA test suite.

For every test kernel this script:
  1. builds random inputs and computes the expected answer with NumPy
  2. runs the kernel on the IR interpreter     (minicuda --run) at -O0 and -O2
  3. compiles it to PTX at -O0 and -O2, checks the PTX and runs it on the
     PTX simulator (tools/ptxsim.py)
  4. with --gpu: also runs the PTX on a real NVIDIA GPU via gpu/ptxrun
     (use this on Google Colab)
and compares every result with the expected answer.

It also runs "negative" tests that must fail with a good error message, and
prints a table of PTX instruction counts at -O0 vs -O2.

    python3 tests/run_tests.py            # interpreter + PTX simulator
    python3 tests/run_tests.py --gpu      # + real GPU (needs gpu/ptxrun built)
    python3 tests/run_tests.py -k matmul  # only tests whose name contains "matmul"
"""
import argparse
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINICUDA = os.path.join(ROOT, "minicuda")
PTXSIM = os.path.join(ROOT, "tools", "ptxsim.py")
PTXRUN = os.path.join(ROOT, "gpu", "ptxrun")
KDIR = os.path.join(ROOT, "tests", "kernels")
OUT = os.path.join(ROOT, "tests", "out")
rng = np.random.default_rng(1234)
DYNAMIC = {}   # test name -> {"-O0": executed PTX instructions, "-O2": ...}


# ------------------------------------------------------------ helpers ---
def write_args(path, args):
    with open(path, "w") as f:
        for a in args:
            if isinstance(a, int):
                f.write(f"i32 {a}\n")
            elif isinstance(a, float):
                f.write("f32 %.9g\n" % np.float32(a))
            elif isinstance(a, np.ndarray):
                ty = "f32" if a.dtype == np.float32 else "i32"
                vals = " ".join(("%.9g" % v) if ty == "f32" else str(int(v)) for v in a.ravel())
                f.write(f"buf {ty} {a.size} {vals}\n")
            else:
                raise TypeError(a)


def read_out(path):
    bufs = {}
    toks = open(path).read().split()
    i = 0
    while i < len(toks):
        assert toks[i] == "buf"
        idx, ty, n = int(toks[i + 1]), toks[i + 2], int(toks[i + 3])
        vals = toks[i + 4: i + 4 + n]
        bufs[idx] = np.array([float(v) for v in vals], dtype=np.float32) if ty == "f32" else \
            np.array([int(v) for v in vals], dtype=np.int32)
        i += 4 + n
    return bufs


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def dimstr(d):
    return ",".join(str(x) for x in d)


class Case:
    def __init__(self, name, file, kernel, grid, block, args, expect, rtol=1e-5, atol=1e-5):
        self.name, self.file, self.kernel = name, file, kernel
        self.grid, self.block, self.args = grid, block, args
        self.expect = expect          # {arg_index: expected array}
        self.rtol, self.atol = rtol, atol


def f32(a):
    return np.asarray(a, dtype=np.float32)


# ------------------------------------------------------------ test cases ---
def make_cases():
    cases = []

    n = 1000
    x, y = f32(rng.standard_normal(n)), f32(rng.standard_normal(n))
    a = 2.5
    cases.append(Case("saxpy", "saxpy.cu", "saxpy", (4,), (256,), [n, a, x, y.copy()],
                      {3: f32(np.float32(a) * x + y)}))

    p, q = f32(rng.standard_normal(777)), f32(rng.standard_normal(777))
    cases.append(Case("vector_add", "vector_add.cu", "vector_add", (7,), (128,),
                      [p, q, np.zeros(777, np.float32), 777], {2: p + q}))

    r = f32(rng.standard_normal(500))
    cases.append(Case("relu", "relu_clamp.cu", "relu", (2,), (256,), [r.copy(), 500], {0: np.maximum(r, 0)}))
    cases.append(Case("clamp", "relu_clamp.cu", "clamp", (4,), (128,),
                      [r, np.zeros(500, np.float32), 500, -0.5, 0.75], {1: np.clip(r, -0.5, 0.75)}))

    g = f32(rng.standard_normal(3000))
    cases.append(Case("grid_stride", "grid_stride.cu", "scale_grid_stride", (3,), (64,), [g.copy(), 3000, 1.5],
                      {0: f32(g * np.float32(1.5) + np.float32(1.0))}))

    N = 24
    A, B = f32(rng.standard_normal((N, N))), f32(rng.standard_normal((N, N)))
    cases.append(Case("matmul", "matmul.cu", "matmul", (2, 2), (16, 16),
                      [A.ravel(), B.ravel(), np.zeros(N * N, np.float32), N], {2: (A @ B).ravel()}, rtol=1e-4, atol=1e-4))

    N = 32
    A, B = f32(rng.standard_normal((N, N))), f32(rng.standard_normal((N, N)))
    cases.append(Case("matmul_tiled", "matmul_tiled.cu", "matmul_tiled", (2, 2), (16, 16),
                      [A.ravel(), B.ravel(), np.zeros(N * N, np.float32), N], {2: (A @ B).ravel()}, rtol=1e-4, atol=1e-4))

    n = 5000
    u, v = f32(rng.standard_normal(n)), f32(rng.standard_normal(n))
    cases.append(Case("dot_product", "dot_product.cu", "dot", (4,), (256,),
                      [u, v, np.zeros(1, np.float32), n], {2: f32([np.dot(u.astype(np.float64), v)])},
                      rtol=1e-4, atol=1e-3))

    data = rng.integers(-1000, 1000, size=2000).astype(np.int32)
    bins = 16
    hist = np.zeros(bins, np.int32)
    for d in data:
        hist[int(d) % bins] += 1  # Python % is already non-negative
    cases.append(Case("histogram", "histogram.cu", "histogram", (8,), (256,),
                      [data, np.zeros(bins, np.int32), 2000, bins], {1: hist}))

    vals = np.array([0, 1, -1, 7, -7, 13, 27, -100, -101, 2**20, -2**20, 99999, 3, -12345, 5, 64], np.int32)
    exp_i, exp_f = [], []
    for xv in vals:
        xv = int(xv)
        if xv == -12345:
            exp_i += [0, 0, 0, 0]; exp_f.append(0.0); continue
        steps, c = 0, abs(xv) + 1
        while c != 1:
            if steps >= 200: break
            c = c >> 1 if c & 1 == 0 else 3 * c + 1
            steps += 1
        odd = 1 + 3 + 5 + 7 + 9
        m = ((xv << 3) ^ ((xv | 5) & 255))
        m = (m + 2**31) % 2**32 - 2**31
        q = int(xv / 7)  # C truncation
        d = q - (xv - q * 7)
        last = min(xv, 10) + max(xv, -10) + (1 if (xv > 0 or xv < -100) else 0)
        exp_i += [steps, m + odd, d, last]
        exp_f.append(float(np.float32(np.float32(xv) / np.float32(4.0)) + np.float32(int(np.float32(1.75) * np.float32(xv)))))
    cases.append(Case("int_ops", "int_ops.cu", "int_ops", (1,), (32,),
                      [vals, np.zeros(64, np.int32), np.zeros(16, np.float32), 16],
                      {1: np.array(exp_i, np.int32), 2: f32(exp_f)}))

    n = 300
    i = np.arange(n)
    cases.append(Case("constant_fold", "constant_fold.cu", "constant_fold", (2,), (256,),
                      [np.zeros(n, np.float32), n], {0: f32((i * 8).astype(np.float32) * 2)}))

    rows, cols = 37, 19
    M = f32(rng.standard_normal((rows, cols)))
    M[5] = 0.0
    norms = np.sqrt((M.astype(np.float64) ** 2).sum(axis=1, keepdims=True))
    expect = np.where(norms > 0, M / np.where(norms > 0, norms, 1), M)
    cases.append(Case("normalize_rows", "sqrt_norm.cu", "normalize_rows", (2,), (32,),
                      [M.ravel().copy(), rows, cols], {0: f32(expect).ravel()}, rtol=1e-5, atol=1e-6))
    return cases


NEGATIVE = [
    ("undeclared variable", "extern \"C\" __global__ void k(float* a) { a[0] = b; }", "undeclared variable 'b'"),
    ("index non-array", "extern \"C\" __global__ void k(int n) { n[0] = 1; }", "cannot be indexed"),
    ("float index", "extern \"C\" __global__ void k(float* a) { a[1.5f] = 1.0f; }", "array index must be an int"),
    ("modulo on float", "extern \"C\" __global__ void k(float* a) { a[0] = a[1] % 2.0f; }", "needs int operands"),
    ("redeclaration", "extern \"C\" __global__ void k(float* a) { int x = 1; int x = 2; }", "already declared"),
    ("break outside loop", "extern \"C\" __global__ void k(float* a) { break; }", "outside of a loop"),
    ("missing semicolon", "extern \"C\" __global__ void k(float* a) { a[0] = 1.0f }", "expected ';'"),
    ("unknown function", "extern \"C\" __global__ void k(float* a) { a[0] = expf(1.0f); }", "unknown function 'expf'"),
    ("bad character", "extern \"C\" __global__ void k(float* a) { a[0] = 1.0f @ 2; }", "unexpected character '@'"),
    ("device function", "float helper(float x) { return x; }", "only __global__ kernels"),
]


# --------------------------------------------------------------- driver ---
def compare(got, case, label):
    for idx, exp in case.expect.items():
        g = got.get(idx)
        if g is None or g.shape != exp.shape:
            return f"{label}: buffer {idx} missing or wrong size"
        if exp.dtype == np.int32:
            if not np.array_equal(g, exp):
                bad = np.nonzero(g != exp)[0][0]
                return f"{label}: buffer {idx}[{bad}] = {g[bad]}, expected {exp[bad]}"
        elif not np.allclose(g, exp, rtol=case.rtol, atol=case.atol):
            bad = np.nonzero(~np.isclose(g, exp, rtol=case.rtol, atol=case.atol))[0][0]
            return f"{label}: buffer {idx}[{bad}] = {g[bad]!r}, expected {exp[bad]!r}"
    return None


def run_case(case, use_gpu):
    src = os.path.join(KDIR, case.file)
    argf = os.path.join(OUT, case.name + ".args")
    write_args(argf, case.args)
    errors = []
    grid, block = dimstr(case.grid), dimstr(case.block)
    for lvl in ("-O0", "-O2"):
        outf = os.path.join(OUT, f"{case.name}{lvl}.interp.out")
        rc, _, err = run([MINICUDA, src, lvl, "--run", case.kernel, "--grid", grid, "--block", block,
                          "--args", argf, "--out", outf])
        if rc != 0:
            errors.append(f"interpreter {lvl} failed: {err.strip()}")
        else:
            e = compare(read_out(outf), case, f"interpreter {lvl}")
            if e: errors.append(e)

        ptx = os.path.join(OUT, f"{case.name}{lvl}.ptx")
        extra = ["--lineinfo"] if lvl == "-O2" else []
        rc, _, err = run([MINICUDA, src, lvl, "-o", ptx] + extra)
        if rc != 0:
            errors.append(f"compile {lvl} failed: {err.strip()}")
            continue
        outf = os.path.join(OUT, f"{case.name}{lvl}.ptxsim.out")
        rc, _, err = run([sys.executable, PTXSIM, ptx, "--kernel", case.kernel, "--grid", grid,
                          "--block", block, "--args", argf, "--out", outf, "--count"])
        for line in err.splitlines():
            if line.startswith("executed "):
                DYNAMIC.setdefault(case.name, {})[lvl] = int(line.split()[1])
        if rc != 0:
            errors.append(f"ptxsim {lvl} failed: {err.strip()}")
        else:
            e = compare(read_out(outf), case, f"ptxsim {lvl}")
            if e: errors.append(e)
        if use_gpu:
            outf = os.path.join(OUT, f"{case.name}{lvl}.gpu.out")
            rc, _, err = run([PTXRUN, ptx, case.kernel, grid, block, argf, outf])
            if rc != 0:
                errors.append(f"GPU {lvl} failed: {err.strip()}")
            else:
                e = compare(read_out(outf), case, f"GPU {lvl}")
                if e: errors.append(e)
    return errors


def ptx_count(src, level):
    rc, _, err = run([MINICUDA, src, level, "--stats", "-o", os.devnull])
    for line in err.splitlines():
        if line.startswith("PTX instructions:"):
            return int(line.split(":")[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also run on a real GPU via gpu/ptxrun")
    ap.add_argument("-k", default="", help="only run tests whose name contains this")
    o = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(MINICUDA):
        sys.exit("build the compiler first: make")
    if o.gpu and not os.path.exists(PTXRUN):
        sys.exit("build gpu/ptxrun first (see gpu/README.md)")

    passed = failed = 0
    for case in make_cases():
        if o.k not in case.name:
            continue
        errs = run_case(case, o.gpu)
        if errs:
            failed += 1
            print(f"FAIL  {case.name}")
            for e in errs:
                print(f"        {e}")
        else:
            passed += 1
            print(f"PASS  {case.name}")

    for name, src, msg in NEGATIVE:
        if o.k not in name and o.k:
            continue
        path = os.path.join(OUT, "neg.cu")
        open(path, "w").write(src)
        rc, _, err = run([MINICUDA, path])
        if rc != 0 and msg in err:
            passed += 1
            print(f"PASS  error: {name}")
        else:
            failed += 1
            print(f"FAIL  error: {name}: expected '{msg}', got rc={rc} {err.strip()}")

    print("\nPTX instruction count   -O0 -> -O2")
    total0 = total2 = 0
    for f in sorted(os.listdir(KDIR)):
        c0, c2 = ptx_count(os.path.join(KDIR, f), "-O0"), ptx_count(os.path.join(KDIR, f), "-O2")
        total0 += c0; total2 += c2
        print(f"  {f:22s} {c0:4d} -> {c2:4d}   ({100 * (c0 - c2) / c0:4.1f}% fewer)")
    print(f"  {'TOTAL':22s} {total0:4d} -> {total2:4d}   ({100 * (total0 - total2) / total0:4.1f}% fewer)")

    if DYNAMIC:
        print("\nExecuted PTX instructions (whole grid, PTX simulator)   -O0 -> -O2")
        t0 = t2 = 0
        for name, d in DYNAMIC.items():
            if "-O0" in d and "-O2" in d:
                t0 += d["-O0"]; t2 += d["-O2"]
                print(f"  {name:22s} {d['-O0']:9d} -> {d['-O2']:9d}   ({100 * (d['-O0'] - d['-O2']) / d['-O0']:4.1f}% fewer)")
        if t0:
            print(f"  {'TOTAL':22s} {t0:9d} -> {t2:9d}   ({100 * (t0 - t2) / t0:4.1f}% fewer)")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
