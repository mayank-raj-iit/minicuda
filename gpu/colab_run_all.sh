#!/usr/bin/env bash
# Run this on Google Colab (Runtime -> Change runtime type -> T4 GPU):
#   !git clone https://github.com/<you>/minicuda && cd minicuda && bash gpu/colab_run_all.sh
set -e
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
make -j2
# Driver API runner. nvcc adds the CUDA stub-library folder, so -lcuda links.
nvcc -O2 -o gpu/ptxrun gpu/ptxrun.cpp -lcuda || \
  g++ -O2 -std=c++17 -o gpu/ptxrun gpu/ptxrun.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda

echo "== 1. full test suite on the real GPU =="
python3 tests/run_tests.py --gpu

echo "== 2. offline assembly with ptxas: registers used per kernel =="
mkdir -p tests/out
./minicuda tests/kernels/matmul_tiled.cu -o tests/out/tiled.ptx
ptxas -arch=sm_75 -v tests/out/tiled.ptx -o tests/out/tiled.cubin

echo "== 3. the real machine code (SASS) ptxas produced =="
cuobjdump -sass tests/out/tiled.cubin | head -60

echo "== 4. compare with NVIDIA's own compiler =="
nvcc -arch=sm_75 -ptx tests/kernels/saxpy.cu -o tests/out/saxpy_nvcc.ptx
./minicuda tests/kernels/saxpy.cu -o tests/out/saxpy_minicuda.ptx
diff <(grep -v '^//' tests/out/saxpy_nvcc.ptx) <(grep -v '^//' tests/out/saxpy_minicuda.ptx) || true
