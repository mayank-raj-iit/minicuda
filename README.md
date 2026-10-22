# MiniCUDA — an optimizing compiler from a CUDA C subset to NVIDIA PTX

MiniCUDA compiles GPU kernels written in a subset of CUDA C into **PTX**, the
virtual instruction set that every NVIDIA GPU compiler targets. The PTX it
writes is loaded by the CUDA driver (which JIT-compiles it to machine code)
or assembled offline by `ptxas`.

```
 kernel.cu ──► Lexer ──► Parser ──► Semantic ──► IR gen ──► Optimizer ──► PTX emitter ──► kernel.ptx
              tokens      AST       checks      3-address    7 passes       .reg/.param       │
                                    + types     code + CFG                  ld/st/setp/bra    ▼
                                                                                    CUDA driver JIT / ptxas
                                                                                    ──► SASS on the GPU
```

Written in C++17 with no dependencies (~3,800 lines of C++, plus a Python test harness and PTX simulator).

## What it supports

```cuda
extern "C" __global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```

* `__global__` kernels with `int`, `float`, `int*`, `float*` parameters
* local `int`/`float` variables, `if`/`else`, `for`, `while`, `break`, `continue`, `return`
* all C arithmetic, comparison, bitwise and short-circuit logical operators; casts
* `threadIdx`, `blockIdx`, `blockDim`, `gridDim` (`.x/.y/.z`)
* `__shared__` arrays and `__syncthreads()`
* `atomicAdd` (int and float, global and shared), `sqrtf`, `fabsf`, `fminf`, `fmaxf`, `min`, `max`, `abs`

Test kernels are valid CUDA, so `nvcc -ptx` compiles the same files for comparison.

## The pipeline

| Phase | File | What it does |
|---|---|---|
| Lexer | `src/lexer.cpp` | characters → tokens, with line/column for error messages |
| Parser | `src/parser.cpp` | recursive descent + precedence climbing → AST |
| Semantic analysis | `src/sema.cpp` | scopes, symbol resolution, type checking, implicit int→float / value→bool conversions |
| IR generation | `src/irgen.cpp` | AST → typed three-address code in basic blocks (a CFG); short-circuit `&&`/`||` |
| Optimizer | `src/opt.cpp` | see below |
| PTX emitter | `src/ptxgen.cpp` | operand legalization, register classes `%r %f %rd %p`, labels, fall-through branches, `.loc` line info |
| Interpreter | `src/interp.cpp` | runs IR for a whole grid on the CPU, with shared memory, barriers and out-of-bounds checking |

### Optimization passes (`-O2`)

1. **CFG simplification** – constant branches become jumps, unreachable blocks are deleted, single-predecessor blocks are merged, jump-only blocks are bypassed.
2. **Local value numbering** – constant folding, constant & copy propagation, algebraic simplification (`x*1`, `x+0`), strength reduction (`x*8 → x<<3`), common-subexpression elimination, redundant-load elimination and store-to-load forwarding.
3. **Global copy & constant propagation** – removes copies whose source and destination are each assigned once, and replaces a variable assigned a constant once with that constant (checked with dominators).
4. **Dead-code elimination** – liveness dataflow analysis over the CFG.
5. **Loop-invariant code motion** – dominator tree → natural loops → hoist invariant computations into a preheader.
6. **FMA / MAD contraction** – `a*b+c` → `fma.rn.f32` / `mad.lo.s32` (nvcc's default `--fmad=true`; disable with `--no-fmad`).
7. **Copy coalescing** – `t = a+b; x = t` → `x = a+b`.

Measured on the 12 test kernels (`make test`):

| | `-O0` | `-O2` | |
|---|---|---|---|
| static PTX instructions | 569 | 443 | 22.1% fewer |
| executed PTX instructions (whole grid) | 1,448,439 | 1,059,418 | 26.9% fewer |

## Build and use

```bash
make                                    # builds ./minicuda (needs g++ or clang++ with C++17)
./minicuda tests/kernels/saxpy.cu -o saxpy.ptx           # compile (-O2 by default)
./minicuda tests/kernels/saxpy.cu -O0 --dump-ir          # look at the IR before optimization
./minicuda tests/kernels/matmul.cu --stats -o /dev/null  # what each pass did
./minicuda tests/kernels/dot_product.cu --lineinfo       # PTX with .loc source lines
```

Run a kernel on the CPU interpreter (no GPU needed):

```bash
cat > args.txt <<EOF
i32 4
f32 2.0
buf f32 4  1 2 3 4
buf f32 4  10 20 30 40
EOF
./minicuda tests/kernels/saxpy.cu --run saxpy --grid 1 --block 4 --args args.txt
```

## Testing

`make test` runs `tests/run_tests.py`, which for 12 kernel tests:

1. computes the expected answer with NumPy,
2. runs the kernel on the IR interpreter at `-O0` and `-O2`,
3. compiles to PTX and runs it on `tools/ptxsim.py`, a small PTX checker and simulator
   (register declarations, register class vs. instruction type, labels, parameters),
4. with `--gpu`, also runs the PTX on a real NVIDIA GPU through `gpu/ptxrun`,

plus 10 negative tests that must fail with the right error message.

### On a real GPU (Google Colab, free T4)

```bash
bash gpu/colab_run_all.sh
```

builds `gpu/ptxrun` (CUDA **Driver API**: `cuModuleLoadDataEx` JIT-compiles the PTX,
`cuLaunchKernel` runs it), runs the whole test suite on the GPU, assembles a kernel
with `ptxas -v` to show register usage, dumps the SASS with `cuobjdump`, and diffs
MiniCUDA's PTX against `nvcc -ptx` for the same source.

## Limitations

No device functions, structs, local arrays, pointer arithmetic, `double`, or
unsigned types. Float literals are always single precision (`2.5` is treated as `2.5f`).
Register allocation is left to `ptxas`, as it is for nvcc's PTX.
