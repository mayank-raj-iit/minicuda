// Per-row L2 normalisation of a matrix (rows x cols), one thread per row.
extern "C" __global__ void normalize_rows(float* m, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
        float ss = 0.0f;
        for (int c = 0; c < cols; c++) {
            float v = m[r * cols + c];
            ss += v * v;
        }
        float norm = sqrtf(ss);
        if (norm > 0.0f) {
            for (int c = 0; c < cols; c++) {
                m[r * cols + c] = m[r * cols + c] / norm;
            }
        }
    }
}
