import numpy as np
from mpi4py import MPI
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

train_df = pd.read_csv('data/mnist_train.csv')
test_df = pd.read_csv('data/mnist_test.csv')
# print("There are ", len(X_train), "images in the training dataset")     
# print("There are ", len(X_test), "images in the test dataset")  

def find_batch(df, size, rank):
    # Convert the features to float32 and normalize using max value of the image arrays.
    X = df.iloc[:, 1:].values.astype('float32') / 255
    # Convert the labels to int
    y = df['label'].values.astype(int)
    # Calculate the size of each chunk
    chunk_size = len(X) // size
    # Determine the start and end indices for slicing
    start_idx = rank * chunk_size
    end_idx = None if rank == size - 1 else (rank + 1) * chunk_size
    # Slice the arrays and return
    return X[start_idx:end_idx], y[start_idx:end_idx]









