import numpy as np
def dot(X, Y):
    """
    Compute the dot product of two arrays X and Y in a sequential manner.
    """
    if X.shape[1] != Y.shape[0]:
        raise ValueError("The number of columns in X must be equal to the number of rows in Y.")
    
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[0]):
                result[i][j] += X[i][k] * Y[k][j]
    return result

# random_matrix1 = np.random.rand(3, 3)
# random_matrix2 = np.random.rand(3, 3)
# assert(np.all(seq_dot(random_matrix1, random_matrix2)) == np.all(np.dot(random_matrix1, random_matrix2)))


def sum(arr, axis=None):
    """
    Sum the elements of an array along a specified axis sequentially.
    """
    if axis is None:
        # Sum over all elements
        total_sum = 0
        for element in arr.flatten():
            total_sum += element
        return total_sum
    else:
        if isinstance(axis, int):
            axis = (axis,)

        for ax in axis:
            if ax >= len(arr.shape):
                raise ValueError("Axis {} is out of bounds for array with {} dimensions".format(ax, arr.ndim))

        # Sum over the specified axes
        sum_arr = arr
        for ax in sorted(axis, reverse=True):
            # Initialize the output array with zeros for the current axis
            output_shape = list(sum_arr.shape)
            del output_shape[ax]
            output_shape = tuple(output_shape)

            new_sum_arr = np.zeros(output_shape, dtype=sum_arr.dtype)

            # Iterate over the array and compute the sum for the current axis
            for index, _ in np.ndenumerate(new_sum_arr):
                index_slice = list(index)
                index_slice.insert(ax, slice(None))
                new_sum_arr[index] = np.sum(sum_arr[tuple(index_slice)])

            sum_arr = new_sum_arr

        return sum_arr


def mean(arr, axis=None):
    if axis is None:
        # Mean over all elements
        total_sum = 0
        num_elements = 0
        for element in arr.flatten():
            total_sum += element
            num_elements += 1
        return total_sum / num_elements
    else:
        # Mean over a specific axis
        # Initialize the output array with zeros
        mean_arr = np.zeros(arr.shape[:axis] + arr.shape[axis+1:], dtype=float)
        
        # Using np.ndindex to correctly index multi-dimensional arrays
        for index in np.ndindex(mean_arr.shape):
            slicing_index = index[:axis] + (slice(None),) + index[axis:]
            mean_arr[index] = np.mean(arr[slicing_index])
        
        return mean_arr

def prod(arr):
    """
    Compute the product of the elements of an array sequentially.
    """
    total_product = 1
    for element in arr:
        total_product *= element
    return total_product

def argmax(arr):
    """
    Find the index of the maximum element in an array sequentially.
    """
    max_index = 0
    for i, element in enumerate(arr):
        if element > max_value:
            max_value = element
            max_index = i
    return max_index

def exp(arr):
    result = np.zeros_like(arr, dtype=float)
    # Flatten the array to iterate over it
    arr_flat = arr.flatten()
    result_flat = result.flatten()
    for i in range(arr_flat.shape[0]):
        result_flat[i] = np.exp(arr_flat[i])
    return result_flat.reshape(arr.shape)

def log(arr):
    result = np.zeros_like(arr, dtype=float)
    # Flatten the array to iterate over it
    arr_flat = arr.flatten()
    result_flat = result.flatten()
    for i in range(arr_flat.shape[0]):
        result_flat[i] = np.log(arr_flat[i])
    return result_flat.reshape(arr.shape)