import numpy as np
from mpi4py import MPI


# warning: can only handle first 3x3 then 2x2 pooling with stride 1
def divide_data(data, num_workers) -> np.ndarray:
    """calculate the subset of data for each worker

    Args:
        data (nd array): input data from the dataset(maybe a csv file)
        num_workers (int): number of computing nodes in the cluster

    Returns:
        array: a list of data subsets corresponding to the workload of each worker
    """

    # return splits
    rows_per_worker = data.shape[0] // num_workers
    redundant_rows = data.shape[0] % num_workers
    splits = []

    for i in range(num_workers):
        start_row = i * rows_per_worker + min(i, redundant_rows)
        end_row = start_row + rows_per_worker + (1 if i < redundant_rows else 0)
        splits.append(data[start_row:end_row])

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
    # print(data)
    print(split)
    partial_conv = model.conv_layers[0].forward(split) ## Conv2d layer
    partial_res = model.conv_layers[2].forward(partial_conv) ## MaxPool2 layer

    # flatten the data
    # print(partial_res.shape)
    partial_flatten = partial_res.flatten() ## Flatten layer

    # use allgatherv to collect the data from each process
    # calculate the size of the data to be received
    recv_size = np.zeros(comm.Get_size(), dtype=int)
    recv_size[comm.Get_rank()] = partial_flatten.shape[0]
    comm.Allgather([recv_size, MPI.INT], [recv_size, MPI.INT])
    # calculate the displacement of the data to be received
    displ = np.zeros(comm.Get_size(), dtype=int)
    displ[1:] = np.cumsum(recv_size)[:-1]
    # allocate a buffer to hold the received data
    recvbuf = np.zeros(np.sum(recv_size), dtype=np.float64)
    # perform the allgatherv
    comm.Allgatherv([partial_flatten, MPI.DOUBLE], [recvbuf, (recv_size, displ), MPI.DOUBLE])
    # reshape the data to be a 1d array
    recvbuf = recvbuf.reshape(-1)
    return recvbuf

# assume that before calling the function, each process has already had a piece of data
# assume all_data has size (num_batch, 10, 32)
def ring_all_reduce(all_data, comm, rank, size):

    # data buffer for receiving data
    
    # separate data into size batches
    n = np.prod(all_data.shape) # total number of parameters
    assert(n % size == 0) # make sure the number of parameters is divisible by the number of processes, can be changed later
    minibatch_size = n // size
    collection = np.reshape(all_data.copy(), (size, minibatch_size))
    recv = np.zeros(minibatch_size, dtype=all_data.dtype)

    # 1. aggregation stage
    for i in range(size - 1):
        send_to = (rank + 1) % size
        recv_from = (rank - 1) % size

        # send data to the next process non-blocking
        req = comm.Isend(collection[(rank - i) % size], dest=send_to, tag=i)

        # receive data from the previous process blocking
        comm.Recv(recv, source=recv_from, tag=i)

        # add the received data to the corresponding position to the local data
        collection[(rank - i - 1) % size] = collection[(rank - i - 1) % size] + recv
        

    # 2. boardcast stage: each worker send one slice of aggregated parameters to the next worker; repeat N times
    recv = collection[(rank + 1) % size].copy()
    for i in range(size - 1):
        send_to = (rank + 1) % size
        recv_from = (rank - 1) % size

        # send data to the next process non-blocking
        req = comm.Isend(recv, dest=send_to, tag=i)

        # receive data from the previous process blocking
        comm.Recv(recv, source=recv_from, tag=i)
        pos = (rank - i) % size

        # warning: not sure whether we need to copy the data
        collection[pos] = recv.copy()
    
    # reshape the data to be the same as the original data
    collection = np.reshape(collection, all_data.shape)
    return collection