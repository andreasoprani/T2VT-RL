import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy, norm
from scipy.integrate import quad
import torch
from torch import autograd

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

def normalized_epanechnikov_torch(x, epsilon = 0.00001):
    """Normalized Epanechnikov kernel."""

    if x < 0 or x > 1:
        return 0
    #elif x == 0:
    #    return 2 * (3/4) * (1 - (x + epsilon).pow(2))
    return 2 * (3/4) * (1 - x.pow(2))

def normalized_epanechnikov_weights_torch(samples, _lambda, epsilon = 0.00001):
    
    if _lambda > 1:
        _lambda = 1
    elif _lambda < 0:
        _lambda = 0

    weights = torch.zeros(samples, dtype=torch.float64)
    
    for i in range(samples):
        t_i = (i + 1) / samples
        t = 1-t_i
        #if t == 0:
        #    t = t + epsilon
        x = t / _lambda
        w = normalized_epanechnikov_torch(x)
        #if w != 0:
        #    weights[i] = w / (samples * _lambda)
        #else:
        #    weights[i] = 0
        weights[i] = w / (samples * _lambda)

    # Normalization
    #non_zero_weights = torch.tensor(list(filter(lambda x: x > 0, weights)), dtype=torch.float64)
    #weights = weights / non_zero_weights.sum()
    weights = weights / weights.sum()

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

def shannon_kernel_weights(weights, samples, h, kernel="epanechnikov"):
    
    euc_distances = [distance.euclidean(weights[i], weights[i+1]) for i in range(len(weights)-1)]
    
    s = entropy(euc_distances, base=2)
    l = (1+s)/samples
    
    return temporal_kernel(samples, l, kernel)
    
def avg_kernel_weights(weights, samples, h, kernel="epanechnikov"):
    
    euc_distances = [distance.euclidean(weights[i], weights[i+1]) for i in range(len(weights)-1)]

    d = euc_distances[-1]
    m = np.average(euc_distances)
    l = (1 + 4 * m / d) / samples
    
    return temporal_kernel(samples, l, kernel)

def timelag_kernel_weights(weights, samples, h, kernel="epanechnikov"):
    
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

def timelag_softmax_kernel_weights(weights, samples, h, kernel="epanechnikov"):
    
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

def likelihood_kernel_weights(weights, samples, h, learning_rate=0.01, kernel="epanechnikov"):
    
    def compute_spatial_kernel(qs, q, h):
        
        xs = torch.zeros(len(qs), dtype=torch.float64)
        for i in range(len(xs)):
            xs[i] = torch.norm((torch.tensor(q, dtype=torch.float64) - torch.tensor(qs[i], dtype=torch.float64)) / h)
        
        ks = torch.tensor(list(map(norm.pdf, xs)), dtype=torch.float64)
        dim = len(q)
        
        return ks / (h**dim)
    
    def compute_loglikelihood(l):
        
        likelihood_components = torch.zeros(len(weights) - 1, dtype=torch.float64)
        
        for x in range(1, len(weights)):
            qs = weights[0:x]
            q = weights[x]
            k_t = normalized_epanechnikov_weights_torch(len(qs), l)
            k_s = compute_spatial_kernel(qs, q, h)
            likelihood_components[x-1] = torch.dot(k_t, k_s)
            
        return torch.log(torch.prod(likelihood_components))
        
    def gradient_ascent_step(l):
        
        l_torch = torch.tensor(l, dtype=torch.float64, requires_grad = True)
        
        with autograd.detect_anomaly():
            loglikelihood = compute_loglikelihood(l_torch)
        
            loglikelihood.backward(l_torch)
            grad = l_torch.grad
            grad = float(grad)
        
        l += learning_rate * grad
        return np.clip(l, 1/samples, 1)
    
    def gradient_ascent_loop(l, max_iter=50, epsilon=0.01):
        i = 0
        while True:
            i += 1
            old_l = l
            l = gradient_ascent_step(l)
            #print("I:", i, "lambda:", l)
            if max_iter is not None and i > max_iter:
                break
            elif epsilon is not None and np.abs(l - old_l) < epsilon:
                break
        
        return l
        
    ls = np.linspace(1/samples, 1, num = 20)
    ls = [gradient_ascent_loop(l) for l in ls]
    ls = [torch.tensor(l) for l in ls]
    l = float(max(ls, key=compute_loglikelihood))
    
    print(l)
    
    return temporal_kernel(samples, l, kernel)

def crossval_kernel_weights(weights, samples, h, kernel="epanechnikov"):
    
    def epanechnikov(x):
        if x < -1 or x > 1:
            return 0
        else:
            return (3/4) * (1 - x**2)
    
    def epanechnikov_integral(low, high):

        F = lambda x : - (x * (x ** 2 - 3)) / 4

        return F(high) - F(low)

    def f(x0, t0, t1, t2, t3, t4):
        """Function to be integered

        Arguments:
            x0 {float} -- t
            t0 {float} -- lambda
            t1 {float} -- t_i
            t2 {float} -- t_j
            t3 {float} -- lower boundary
            t4 {float} -- upper boundary
        """
        
        t = x0
        l = t0
        t_i = t1
        t_j = t2
        lower_boundary = t3
        upper_boundary = t4
        
        out = epanechnikov((t - t_i)/l) * epanechnikov((t - t_j)/l)

        if t < l:
            out /= (epanechnikov_integral(-1, (lower_boundary + t)/l))**2
        elif t > upper_boundary - l:
            out /= (epanechnikov_integral(- (upper_boundary - t) / l, 1))**2
        
        return out 
    
    def integral_calculation(l):

        components = np.zeros((samples, samples), dtype=np.float64)
        
        lower_boundary = 1 / samples
        upper_boundary = 1

        for i in range(samples):
            for j in range(samples):
                c = np.e ** (-(np.linalg.norm(weights[i] - weights[j]) ** 2 / (4 * h ** 2)))
                integral, err = quad(f, lower_boundary, upper_boundary, args=(l, i/samples, j/samples, lower_boundary, upper_boundary))
                components[i, j] = c * integral

        dim = len(weights[0])

        out = np.sum(components)
        out /= samples ** 2 * l ** 2 * h ** (2 * dim)
        out /= (4 * np.pi * h ** 2) ** (dim/2)

        return out
    
    def estimator_without_i(i, l):
        
        t_i = i/samples
        q_i = np.array(weights[i], dtype=np.float64)
        
        temporal_components = np.zeros(samples - 1, dtype=np.float64)
        spatial_components = np.zeros(samples - 1, dtype=np.float64)
        
        index = 0
        
        for j in range(samples):
            
            if j == i:
                continue
            
            q_j = np.array(weights[j], dtype=np.float64)
            
            temporal_components[index] = epanechnikov(np.abs(i-j)/(samples*l))
            spatial_components[index] = norm.pdf(np.linalg.norm(q_i - q_j)/h)
            
            index += 1

        temporal_components = [t / np.sum(temporal_components) for t in temporal_components]

        dim = len(q_i)
        
        return np.dot(temporal_components, spatial_components) / (samples * l * (h**dim))

    def calculate_cv(l):

        integral_component = integral_calculation(l)
        estimator_component = (2 / samples) * np.sum([estimator_without_i(i, l) for i in range(samples)])

        print("Lambda: {0:4.3f}, IC: {1:8.6f}, EC: {2:8.6f}".format(l, integral_component, estimator_component))

        return integral_component - estimator_component

    epsilon = 0.001
    ls = np.linspace(1/samples + epsilon, 1, num = 20)
    cvs = [calculate_cv(l) for l in ls]
    
    #for l, cv in zip(ls, cvs):
    #    print("Lambda: {0:4.3f}, CV: {1:8.6f}".format(l, cv))
       
    l = ls[np.argmin(cvs)]
    
    print("Selected lambda:", l)

    return temporal_kernel(samples, l, kernel)

presets = {
    "shannon": shannon_kernel_weights,
    "avg": avg_kernel_weights,
    "timelag": timelag_kernel_weights,
    "timelag_softmax": timelag_softmax_kernel_weights,
    "likelihood": likelihood_kernel_weights,
    "crossval": crossval_kernel_weights
}

def temporal_weights_calculator(weights, samples, preset="fixed", _lambda=1, h=1, kernel="epanechnikov"):
    
    if preset == "fixed":
        return temporal_kernel(samples, _lambda, kernel)
    
    if preset not in presets:
        print("Preset not found.")
        return None
    
    return presets[preset](weights, samples, h, kernel=kernel)
