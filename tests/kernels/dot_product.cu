// Dot product: shared-memory tree reduction per block, then one atomicAdd per block.
// blockDim.x must be 256.
extern "C" __global__ void dot(const float* a, const float* b, float* result, int n) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    float temp = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        temp += a[i] * b[i];
    }
    cache[tid] = temp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s = s / 2) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&result[0], cache[0]);
    }
}
