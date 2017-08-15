import scipy as sp

def average(data):
    """
    Calculate the average of the given data.
    :param data: A 1D array of floats to be averaged.
    :return: The average.
    """
    counts = len(data)
    total = sum(data)
    return total / counts

def stddev(data):
    """
    Calculate the standard deviation for the given data.
    :param data: A 1D array containing the data.
    :return: The standard deviation.
    """
    counts = len(data)
    ave = average(data)
    total = sum(data * data)
    return (total - ave**2) / counts