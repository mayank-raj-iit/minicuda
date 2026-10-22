// Integer atomics: count how many values fall in each of `bins` buckets.
extern "C" __global__ void histogram(const int* data, int* hist, int n, int bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = data[i];
        int b = v % bins;
        if (b < 0) b += bins;
        atomicAdd(&hist[b], 1);
    }
}
