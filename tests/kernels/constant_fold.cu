// Lots of work the optimizer can do at compile time.
extern "C" __global__ void constant_fold(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = 16 * 4;          // folded to 64
    int unused = width * 1000;   // dead: removed by DCE
    float scale = 2.0f * 0.5f;   // folded to 1.0, then x * 1.0 -> x
    if (width > 100) {           // always false: branch and block deleted
        out[i] = -1.0f;
    }
    if (i < n) {
        out[i] = (float)(i * 8) * scale + (float)(i * 8);   // i*8 computed once (CSE), *8 -> <<3
    }
}
