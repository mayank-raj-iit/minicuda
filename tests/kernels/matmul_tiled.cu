// Tiled matrix multiply with shared memory (TILE = 16). N must be a multiple of 16.
extern "C" __global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[256];
    __shared__ float Bs[256];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float sum = 0.0f;
    for (int t = 0; t < N / 16; t++) {
        As[ty * 16 + tx] = A[row * N + t * 16 + tx];
        Bs[ty * 16 + tx] = B[(t * 16 + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < 16; k++) {
            sum += As[ty * 16 + k] * Bs[k * 16 + tx];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
