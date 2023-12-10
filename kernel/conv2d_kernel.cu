extern "C" {
__global__ void conv2d_kernel(float *input, float *filters, float *bias, float *output, 
                              int kernel_height, int kernel_width, int out_height, int out_width, 
                              int n_filters, int n_rows, int n_cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < out_height && j < out_width && k < n_filters) {
        float sum = 0;
        for (int di = 0; di < kernel_height; ++di) {
            for (int dj = 0; dj < kernel_width; ++dj) {
                sum += input[(i + di) * n_cols + j + dj] * filters[di * kernel_width + dj + k * kernel_height * kernel_width];
            }
        }
        output[i * out_width * n_filters + j * n_filters + k] = sum + bias[k];
    }
}
}
