extern "C" {

__global__ void backprop_kernel(float *d_L_d_out, float *last_input, float *d_L_d_filters, 
                                float *d_L_d_bias, int kernel_height, int kernel_width, 
                                int n_rows, int n_cols, int num_filters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < n_rows - kernel_height + 1 && j < n_cols - kernel_width + 1 && f < num_filters) {
        float region_product = 0;
        for (int di = 0; di < kernel_height; ++di) {
            for (int dj = 0; dj < kernel_width; ++dj) {
                region_product = last_input[(i + di) * n_cols + j + dj];
                atomicAdd(&d_L_d_filters[di * kernel_width + dj + f * kernel_height * kernel_width], d_L_d_out[i * n_cols * num_filters + j * num_filters + f] * region_product);
            }
        }
        atomicAdd(&d_L_d_bias[f], d_L_d_out[i * n_cols * num_filters + j * num_filters + f]);
    }
}
}