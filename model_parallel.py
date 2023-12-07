# Apply model parallelism on fully connected layers.

import numpy as np
from mpi4py import MPI
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import mnist
from cnn_model import Linear

warnings.filterwarnings("ignore")
warnings.warn("this will not show")
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
   

def find_batch(images, labels, size, rank):
    # Convert the features to float32 and normalize using max value of the image arrays.
    X = images.values.astype('float32') / 255
    # Convert the labels to int
    y = labels.values.astype(int)
    # Calculate the size of each chunk
    chunk_size = len(X) // size
    # Determine the start and end indices for slicing
    start_idx = rank * chunk_size
    end_idx = None if rank == size - 1 else (rank + 1) * chunk_size
    # Slice the arrays and return
    return X[start_idx:end_idx], y[start_idx:end_idx]


def distribute_activations(last_stage_activations):
    # Divide the last-stage convolutional layer activities among all workers
    # Each worker sends 1/K of their data to all other workers
    send_buf = np.array_split(last_stage_activations, size)
    recv_buf = np.empty_like(send_buf)
    # distribute the chunks to all processes
    comm.Alltoall(send_buf, recv_buf)
    # Concatenate the received chunks to form the full batch for this worker
    concatenated_activations = np.concatenate(recv_buf)
    return concatenated_activations

def bcast_weights(model):
    # Broadcast the weights of the fully-connected layers to all workers
    for layer in model.layers:
        if isinstance(layer, Linear):
            comm.Bcast(layer.w, root=0)

def agg_gradient(model):
    # Aggregate the gradients of the fully-connected layers from all workers
    for layer in model.layers:
        if isinstance(layer, Linear):
            # Initialize buffer to gather gradients from all workers
            recv_buff = np.empty((size, *layer.dw.shape), dtype=float) if rank == 0 else None
            # Gather gradients from all workers at the root
            comm.Gather(layer.dw, recv_buff, root=0)
            if rank == 0:
                # Sum up gradients from all workers
                layer.dw = np.sum(recv_buff, axis=0)








