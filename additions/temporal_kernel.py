import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy

def normalized_epanechnikov(x):
    """Normalized Epanechnikov kernel."""

    if x < 0 or x > 1:
        return 0
    else:
        return 2 * (3/4) * (1 - x**2)

def normalized_epanechnikov_weights(samples, _lambda):
    """Calculates the weights of the mixture of gaussians using a 
    Normalized Epanechnikov kernel.

    Parameters
    ----------
    samples : int
              number of weights to be calculated.
    _lambda : float
              bandwidth of the kernel.

    Returns
    -------
    the list of weights
    """
    
    # Clipping
    if _lambda > 1:
        _lambda = 1
    elif _lambda < 0:
        _lambda = 0

    weights = np.zeros(samples)

    for i in range(samples):
        t_i = (i + 1) / samples
        weights[i] = normalized_epanechnikov((1-t_i)/_lambda) / (samples * _lambda)

    # Normalization
    weights = weights / np.sum(weights)

    return weights

# Different kernels can be used, they have to be implemented in this file.
kernels = {
    "epanechnikov": normalized_epanechnikov_weights
}

def temporal_kernel(samples, _lambda, kernel = "epanechnikov"):
    """Executes the weights calculation with the requested kernel.
    
    Parameters
    ----------
    samples : int
              number of weights to be calculated.
    _lambda : float
              bandwidth of the kernel.
    kernel : string (default = "epanechnikov")
              the requested kernel.

    Returns
    -------
    the list of weights
    """

    if kernel not in kernels:
        print("Kernel not found.")
        return None

    return kernels[kernel](samples, _lambda)

def shannon_kernel_weights(weights, samples, kernel="epanechnikov"):
    
    euc_distances = [distance.euclidean(weights[i], weights[i+1]) for i in range(len(weights)-1)]
    
    s = entropy(euc_distances, base=2)
    l = (1+s)/samples
    
    return temporal_kernel(samples, l, kernel)
    
def avg_kernel_weights(weights, samples, kernel="epanechnikov"):
    
    euc_distances = [distance.euclidean(weights[i], weights[i+1]) for i in range(len(weights)-1)]

    d = euc_distances[-1]
    m = np.average(euc_distances)
    l = (1 + 4 * m / d) / samples
    
    return temporal_kernel(samples, l, kernel)

def timelag_kernel_weights(weights, samples, kernel="epanechnikov"):
    
    timelag_weights = []
    
    for tau in range(1, len(weights)):
        dists = []
        for i in range(0, len(weights) - tau):
            dists.append(distance.euclidean(weights[i], weights[i + tau]))
        d = np.average(dists)
        w = 1/d
        timelag_weights.append((tau, w))

    w_sum = np.sum([w for (_, w) in timelag_weights])
    weighted_timelags = [tau * w / w_sum for (tau, w) in timelag_weights]
    
    l = np.sum(weighted_timelags) / samples
    
    return temporal_kernel(samples, l, kernel)

def timelag_softmax_kernel_weights(weights, samples, kernel="epanechnikov"):
    
    timelag_weights = []
    
    for tau in range(1, len(weights)):
        dists = []
        for i in range(0, len(weights) - tau):
            dists.append(distance.euclidean(weights[i], weights[i + tau]))
        d = np.average(dists)
        w = np.exp(1/d)
        timelag_weights.append((tau, w))

    w_sum = np.sum([w for (_, w) in timelag_weights])
    weighted_timelags = [tau * w / w_sum for (tau, w) in timelag_weights]
    
    l = np.sum(weighted_timelags) / samples
    
    return temporal_kernel(samples, l, kernel)

presets = {
    "shannon": shannon_kernel_weights,
    "avg": avg_kernel_weights,
    "timelag": timelag_kernel_weights,
    "timelag_softmax": timelag_softmax_kernel_weights
}

def temporal_weights_calculator(weights, samples, preset="fixed", _lambda=1, kernel="epanechnikov"):
    
    if preset == "fixed":
        return temporal_kernel(samples, _lambda, kernel)
    
    if preset not in presets:
        print("Preset not found.")
        return None
    
    return presets[preset](weights, samples, kernel)