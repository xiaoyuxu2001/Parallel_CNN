import numpy as np
import pandas as pd
from tqdm import tqdm
from mpi4py import MPI
from sequential_cnn import CNN



# warning: can only handle first 3x3 then 2x2 pooling with stride 1
def divide_data(data, num_workers) -> np.ndarray:
    """calculate the subset of data for each worker

    Args:
        data (nd array): input data from the dataset(maybe a csv file)
        num_workers (int): number of computing nodes in the cluster

    Returns:
        array: a list of data subsets corresponding to the workload of each worker
    """
    rows_per_worker = data.shape[0] // num_workers
    redundant_rows = data.shape[0] % num_workers

    for i in range(num_workers):
        if i < redundant_rows:
            splits[i] = data[i * (rows_per_worker + 1):((i + 1) * (rows_per_worker + 1) + 2)]
        elif not i == num_workers - 1:
            splits[i] = data[i * rows_per_worker + redundant_rows:((i + 1) - 1 * rows_per_worker + redundant_rows) + 2]
        else:
            splits[i] = data[i * rows_per_worker + redundant_rows - 1:]

    splits = np.array(splits)

    return splits


def data_parallel_forward(data, comm, model):
    """perform data parallelism on the forward pass of the model and return the reduced data
       need initialization of the model and communication before calling this function

    Args:
        data (nd array): input data from the dataset(maybe a csv file)
        comm (MPI communicator): MPI communicator
        model (CNN): CNN model

    Returns:
        array: a list of flattened data for each worker
    """
    # spread the data across the processes
    partition = divide_data(data, comm.Get_size())
    # send the data to each process
    split = comm.scatter(partition, root=0)
    # apply the forward pass on each process
    partial_conv = model.conv.forward(split)
    partial_res = model.pool.forward(partial_conv)

    # flatten the data
    partial_flatten = model.flatten(partial_res)

    # use allgatherv to collect the data from each process
    # calculate the size of the data to be received
    recv_size = np.zeros(comm.Get_size(), dtype=np.int)
    recv_size[comm.Get_rank()] = partial_flatten.shape[0]
    comm.Allgather([recv_size, MPI.INT], [recv_size, MPI.INT])
    # calculate the displacement of the data to be received
    displ = np.zeros(comm.Get_size(), dtype=np.int)
    displ[1:] = np.cumsum(recv_size)[:-1]
    # allocate a buffer to hold the received data
    recvbuf = np.zeros(np.sum(recv_size), dtype=np.float64)
    # perform the allgatherv
    comm.Allgatherv([partial_flatten, MPI.DOUBLE], [recvbuf, (recv_size, displ), MPI.DOUBLE])
    # reshape the data to be a 1d array
    recvbuf = recvbuf.reshape(-1)
    return recvbuf







    


