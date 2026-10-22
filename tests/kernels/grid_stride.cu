// grid-stride loop: each thread handles many elements
extern "C" __global__ void scale_grid_stride(float* data, int n, float s) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        data[i] = data[i] * s + 1.0f;
    }
}
