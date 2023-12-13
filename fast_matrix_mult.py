from mpi4py import MPI
import numpy as np

def matmul_parallel_divideA_horizontal(A, B, rows_A, cols_A, cols_B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Determine the size of each subtask
    rows_per_proc = rows_A // size
    # Initialize local matrices
    local_A = np.empty((rows_per_proc, cols_A), dtype= "float")
    local_C = np.empty((rows_per_proc, cols_B), dtype="float")
    # Scatter rows of A to different processes
    comm.Scatterv(A, local_A, root=0)
    print("here")
    # Broadcast B to all processes
    comm.Bcast(B, root=0)
    # Local computation: Matrix multiplication
    print(local_A)
    local_C = np.dot(local_A, B)
    # Gather the local C matrices back to the root process
    final_C = None
    if rank == 0:
        final_C = np.empty((rows_A, cols_B), dtype="float")
    comm.Gather(local_C, final_C, root=0)
    return final_C



def matmul_parallel_divideB_vertical(A, B, rows_A, cols_A, cols_B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine the size of each subtask
    cols_per_proc = cols_B // size
    # Initialize local matrices
    local_B = np.empty((cols_A, cols_per_proc), dtype= "float")
    local_C = np.empty((rows_A, cols_per_proc), dtype="float")
    print(B.shape)
    print(local_B.shape)
    # Scatter rows of A to different processes
    comm.Scatter(B, local_B, root=0)
    # Broadcast B to all processes
    comm.Bcast(A, root=0)
    local_C = np.dot(A, local_B)
    # Gather the local C matrices back to the root process
    final_C = None
    if rank == 0:
        final_C = np.empty((rows_A, cols_B), dtype="float")
    comm.Gather(local_C, final_C, root=0)
    return final_C

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    A = np.array([[1., 2., 3., 4], [4., 5., 6., 7], [7., 8., 9., 7],[1., 2., 3., 3]])
    B = np.array([[1., 2., 3., 4], [4., 5., 6., 7], [7., 8., 9., 7],[1., 2., 3., 3]])
    C = matmul_parallel_divideB_vertical(A, B, 4, 4 ,4)

    if rank == 0:
        print("Result matrix C:\n", C)

if __name__ == "__main__":
    main()