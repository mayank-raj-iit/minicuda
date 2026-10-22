// if / else, fminf / fmaxf
extern "C" __global__ void relu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (x[i] < 0.0f) {
        x[i] = 0.0f;
    }
}

extern "C" __global__ void clamp(const float* in, float* out, int n, float lo, float hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        if (v < lo) v = lo;
        else if (v > hi) v = hi;
        out[i] = fmaxf(fminf(v, hi), lo);
    }
}
