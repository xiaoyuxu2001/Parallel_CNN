import numpy as np
from mpi4py import MPI
from conv import Conv2d
from maxpool import MaxPool2
from dense import  Relu, Flatten
import numpy as np
import logging

class Parallel_Convolution:
    def ___init__(self, learning_rate):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.layers = [
            Conv2d(num_filters=32, kernel_size=(3, 3), learning_rate=learning_rate),
            Relu(),
            MaxPool2(),
            Flatten(),
        ]
    
    def forward(self, image, epoch):
        '''
        Returns:
        array: a list of data subsets corresponding to the workload of each worker
        '''
        
        # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
        # to work with. This is standard practice.
        # Forward pass through each layer
        out = image / 255  # Normalize input
        for layer in self.layers:  # Exclude last layer (SoftMaxCrossEntropy)
            if isinstance(layer, Conv2d):
                out = layer.forward(out, epoch)
            else:
                out = layer.forward(out)
            if isinstance(layer, MaxPool2):
                logging.debug("after pool", out.shape)
                self.before_flat = out.shape
        return out
    
    def backprop(self, gradient):
        # Backward pass through each layer
        for layer in reversed(self.layers):
            gradient = layer.backprop(gradient)
        return gradient

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
    partial_conv = model.layers[0].forward(split) ## Conv2d layer
    
    partial_res = model.layers[2].forward(partial_conv) ## MaxPool2 layer

    # flatten the data
    partial_flatten = model.layers[2](partial_res) ## Flatten layer

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

# assume that before calling the function, each process has already had a piece of data
def ring_all_reduce(all_data, comm, op):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # data buffer for receiving data
    recv = np.zeros(all_data.shape[1:], dtype=np.float16)
    collection = all_data.copy()
    

    # start aggregation stage
    for i in range(size - 1):
        send_to = (rank + 1) % size
        recv_from = (rank - 1) % size

        # send data to the next process non-blocking
        req = comm.Isend(collection[(rank - i) % size], dest=send_to, tag=i)
        # receive data from the previous process blocking
        # allocate a buffer to hold the received data
        comm.Recv(recv, source=recv_from, tag=i)
        # add the received data to the corresponding position to the local data
        collection[(rank - i - 1) % size] = op(collection[(rank - i - 1) % size], recv)
        

    # start boardcast stage: each worker send one slice of aggregated parameters to the next worker; repeat N times
    recv = np.collection[(rank - 1) % size].copy()
    for i in range(size - 1):
        send_to = (rank + 1) % size
        recv_from = (rank - 1) % size

        # send data to the next process non-blocking
        req = comm.Isend(recv, dest=send_to, tag=i)
        # receive data from the previous process blocking
        # allocate a buffer to hold the received data
        comm.Recv(recv, source=recv_from, tag=i)
        pos = (rank - i - 1) % size
        # warning: not sure whether we need to copy the data
        collection[pos] = recv.copy()
    
    return collection


def gather_batch(batch_split_x: np.ndarray, cur_round:int, comm):
    '''
    This function is used to gather batch data before running model parallelism
    input: batch_split_x: the data in the current process
    '''
    send_data = batch_split_x[0, np.newaxis, :, :] * np.ones((size, 1, 1))
    send_data = batch_split_x[cur_round * 32: (cur_round + 1) * 32].copy()
    size = comm.Get_size()
    recv_size = np.zeros(comm.Get_size(), dtype=np.int)
    recv_size[comm.Get_rank()] = send_data.shape[0]
    recvbuf = [np.zeros(recv_size[i], dtype=np.float64) for i in range(size)]
    comm.Allgatherv(send_data, [recvbuf[i] for i in range(size)])
    return recvbuf









    







    


