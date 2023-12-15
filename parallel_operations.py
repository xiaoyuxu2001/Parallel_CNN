import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule

def dot(X, Y):
    """
    Compute the dot product of two arrays X and Y using CUDA.
    """
    if X.shape[1] != Y.shape[0]:
        raise ValueError("The number of columns in X must be equal to the number of rows in Y.")

    # Allocate memory on GPU
    X_gpu = gpuarray.to_gpu(X.astype(np.float32))
    Y_gpu = gpuarray.to_gpu(Y.astype(np.float32))
    result_gpu = gpuarray.zeros((X.shape[0], Y.shape[1]), np.float32)

    # CUDA kernel
    kernel_code = """
    __global__ void MatrixMulKernel(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < A_rows && col < B_cols) {
            float sum = 0;
            for (int k = 0; k < A_cols; ++k) {
                sum += A[row * A_cols + k] * B[k * B_cols + col];
            }
            C[row * B_cols + col] = sum;
        }
    }
    """

    mod = SourceModule(kernel_code)
    matrix_mul_kernel = mod.get_function("MatrixMulKernel")

    # Grid and block dimensions
    block_size = (16, 16, 1)  # Example values, adjust as needed
    grid_size = (int(np.ceil(Y.shape[1] / block_size[0])), int(np.ceil(X.shape[0] / block_size[1])), 1)

    # Launch kernel
    matrix_mul_kernel(X_gpu, Y_gpu, result_gpu,
                      np.int32(X.shape[0]), np.int32(X.shape[1]), np.int32(Y.shape[1]),
                      block=block_size, grid=grid_size)

    # Copy result back to CPU
    result = result_gpu.get()
    return result