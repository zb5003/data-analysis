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

def chi_squared(measured, theory, sigma):
    """
    Calculate the chi squared value for a given data set and theoretical predictions.
    :param measured: The measured data.
    :param theory: The model predictions for each measured value.
    :param sigma: The measurement uncertainty for each measured value.
    :return: The chi squared value.
    """
    chi = (measured - theory)**2 / sigma
    return sum(chi * chi)

def beta(measured, theory, sigma, model_gradient_component):
    """
    Compute a component of the gradient of chi squared in the direction of a particular model parameter.
    :param measured: The measured data.
    :param theory: The model predictions for each measured value.
    :param sigma: The measurement uncertainty for each measured value.
    :param model_gradient_component: The component of the model gradient in the direction of a particular 
                                     model parameter.
    :return: The gradient in the direction of a particular parameter.
    """
    chi = (measured - theory) / sigma
    return sum(chi * model_gradient_component)

def alpha(sigma, model_gradient_component_i, model_gradient_component_j):
    """
    Compute the (i, j) component of the model Hessian.
    Note that this is not the actual Hessian, but the one typically used in nonlinear model fitting.
    See William Press, Numerical Recipies Third Edition section 15.5.1 (page 800) for more details.
    :param sigma: The measurement errors
    :param model_gradient_component_i: The component of the gradient in the direction of the ith parameter.
    :param model_gradient_component_j: The component of the gradient in the direction of the jth parameter.
    :return: The (i, j) component of the Hessian.
    """
    return sum(model_gradient_component_i * model_gradient_component_j / (sigma * sigma)) / 2
