// Integer arithmetic, bit operations, casts, while loop, break/continue, short-circuit logic.
extern "C" __global__ void int_ops(const int* in, int* out, float* fout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && in[i] != -12345) {          // second test must not run when i >= n
        int x = in[i];
        // Collatz steps (bounded)
        int steps = 0;
        int c = abs(x) + 1;
        while (c != 1) {
            if (steps >= 200) break;
            if ((c & 1) == 0) c = c >> 1;
            else c = 3 * c + 1;
            steps++;
        }
        int odd = 0;
        for (int k = 0; k < 10; k++) {
            if (k % 2 == 0) continue;
            odd += k;
        }
        int m = (x << 3) ^ (x | 5) & 255;
        int d = x / 7 - x % 7;
        out[i * 4 + 0] = steps;
        out[i * 4 + 1] = m + odd;
        out[i * 4 + 2] = d;
        out[i * 4 + 3] = min(x, 10) + max(x, -10) + (x > 0 || x < -100);
        fout[i] = (float)x / 4.0f + (float)(int)(1.75f * (float)x);
    }
}
