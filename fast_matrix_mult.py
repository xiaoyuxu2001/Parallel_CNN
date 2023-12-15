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
    print(local_C)
    # Gather the local C matrices back to the root process
    final_C = None
    if rank == 0:
        final_C = np.empty((rows_A, cols_B), dtype="float")
    comm.Gather(local_C, final_C, root=0)
    return final_C

def matmul_parallel_forward(local_A_chunk, full_B, rows_A, cols_A, cols_B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    cols_per_proc = cols_B // size
    
    local_B = np.empty((cols_A, cols_per_proc), dtype="float")
    local_C = np.empty((local_A_chunk.shape[0], cols_per_proc), dtype="float")

    # Scatter columns of B to different processes
    if rank == 0:
        # Prepare the array of sendcounts & displacements for scattering B
        sendcounts = [cols_per_proc * cols_A] * size
        displacements = [i * cols_per_proc * cols_A for i in range(size)]
        full_B_reshaped = full_B.flatten()
    else:
        sendcounts = displacements = full_B_reshaped = None
    
    comm.Scatterv([full_B_reshaped, sendcounts, displacements, MPI.FLOAT], local_B.flatten(), root=0)

    # Each process already has the local chunk of A, so no need for broadcasting A

    # Local computation: Matrix multiplication
    local_C = np.dot(local_A_chunk, local_B)

    # Gather the local C matrices back to the root process
    final_C = None
    if rank == 0:
        # Prepare the array of recvcounts & displacements for gathering C
        recvcounts = [rows_A * cols_per_proc] * size
        displacements = [i * rows_A * cols_per_proc for i in range(size)]
        final_C = np.empty((rows_A, cols_B), dtype="float")
    
    comm.Gatherv(local_C.flatten(), [final_C, recvcounts, displacements, MPI.FLOAT], root=0)

    return final_C if rank == 0 else None
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