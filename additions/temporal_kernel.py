import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
import torch
  
def epanechnikov(x):
    """Basic calculation of the Epanechnikov function"""

    if x < -1 or x > 1:
        return 0
    else:
        return (3/4) * (1 - x**2)
    
def epanechnikov_integral(low, high):
    """Calculation of the definite integral of the Epanechnikov function on a range.

    Arguments:
        low {float} -- lower bound of the interval.
        high {float} -- higher bound of the integral.

    Returns:
        float -- definite integral of the Epanechnikov function on the given range.
    """
    
    F = lambda x : - (x * (x ** 2 - 3)) / 4

    return F(high) - F(low)

def normalized_epanechnikov(x):
    """Normalized Epanechnikov kernel (for extreme boundary correction)."""

    if x < 0 or x > 1:
        return 0
    else:
        return 2 * (3/4) * (1 - x**2)

def normalized_epanechnikov_weights(samples, l):
    """
    Calculates the weights of the mixture of gaussians using a normalized Epanechnikov kernel.

    Arguments:
        samples {int} -- number of weights to be calculated.
        l {float} -- bandwidth of the kernel.

    Returns:
        [float] -- the list of weights
    """
    
    # Clipping
    if l > 1:
        l = 1
    elif l < 0:
        l = 0

    weights = np.zeros(samples)

    for i in range(samples):
        t_i = (i + 1) / samples
        weights[i] = normalized_epanechnikov((1-t_i)/l) / (samples * l)

    # Normalization
    weights = weights / np.sum(weights)

    return weights

# Different kernels can be used, they have to be implemented in this file.
kernels = {
    "epanechnikov": normalized_epanechnikov_weights
}

def temporal_kernel(samples, l, kernel = "epanechnikov"):
    """ Executes the weights calculation with the requested kernel.

    Arguments:
        samples {int} -- number of weights to be calculated.
        l {gloat} -- bandwidth of the kernel-

    Keyword Arguments:
        kernel {str} -- the requested kernel (default: {"epanechnikov"})

    Returns:
        [float] -- List of weights.
    """

    if kernel not in kernels:
        print("Kernel not found.")
        return None

    return kernels[kernel](samples, l)

def likelihood_kernel_weights(weights, samples, h, optimization="grid-search", learning_rate=0.01, kernel="epanechnikov"):
    """ Calculation of the kernel weights with the likelihood heuristic.

    Arguments:
        weights {[float]} -- a list containing the weights of the past Q functions.
        samples {int} -- number of past Q functions.
        h {float} -- spatial bandwidth

    Keyword Arguments:
        optimization {str} -- optimization method, grid-search or gradient-ascent (default: {"grid-search"})
        learning_rate {float} -- learning rate for gradient ascent (default: {0.01})
        kernel {str} -- the requested temporalkernel (default: {"epanechnikov"})

    Returns:
        [float] -- List of weights.
    """
    
    def compute_temporal_kernel(ts, t, l):
        """ Computation of the list of the temporal kernel components of the estimator.

        Arguments:
            ts {[float]} -- list of time points.
            t {float} -- time point in which the computation has to be made.
            l {float} -- temporal bandwidth of the kernel.

        Returns:
            [float] -- torch tensor of temporal kernel components
        """
        
        xs = torch.zeros(len(ts), dtype=torch.float64)
        
        # Calculation performed for all the points except the time point of interest.
        for i in range(len(ts)):
            if t == ts[i]:
                xs[i] = 0
            else:
                xs[i] = epanechnikov((t-ts[i])/l) / (samples * l)
                
        # Boundary correction
        lower_boundary = 1 / samples
        upper_boundary = 1
        if t < lower_boundary + l:
            xs /= epanechnikov_integral(-1, (t - lower_boundary) / l)
        elif t > upper_boundary - l:
            xs /= epanechnikov_integral((t - upper_boundary) / l, 1)
        
        # Normalization
        xs /= xs.sum()
        
        return xs
    
    def compute_spatial_kernel(qs, q, h):
        """ Computation of the list of the spatial kernel components of the estimator.

        Arguments:
            qs {[[float]]} -- list of the Q functions.
            q {[float]} -- Q function in which the computation has to be made.
            h {float} -- spatial bandwidth of the kernel.

        Returns:
            [float] -- torch tensor of spatial kernel components
        """
        
        xs = torch.zeros(len(qs), dtype=torch.float64)
        
        for i in range(len(xs)):
            xs[i] = torch.norm((torch.tensor(q, dtype=torch.float64) - torch.tensor(qs[i], dtype=torch.float64))) / h
        
        # Application of the gaussian kernel
        ks = torch.tensor(list(map(norm.pdf, xs)), dtype=torch.float64)
        
        dim = len(q)
        
        return ks / (h ** dim)
    
    def compute_loglikelihood(l):
        """Log-likelihood computation

        Arguments:
            l {float} -- temporal bandwidth.

        Returns:
            float -- log-likelihood.
        """
        
        likelihood_components = torch.zeros(samples, dtype=torch.float64)
        
        dim = len(weights[0])
        
        for x in range(samples):
            # temporal components computation
            ts = torch.linspace(1/samples, 1, samples)
            t = (x + 1) / samples
            k_t = compute_temporal_kernel(ts, t, l)
            
            # spatial components computation
            qs = weights
            q = weights[x]
            k_s = compute_spatial_kernel(qs, q, h)
            
            #print("L: {0:4.3f}, I: {1:1d}".format(l, x))
            #print("Kt:", k_t.numpy())
            #print("Ks:", k_s.numpy())
            
            # dot product
            likelihood_components[x] = torch.dot(k_t, k_s)
           
        return torch.log(torch.prod(likelihood_components))
        
    def gradient_ascent_step(l):
        """ Gradient ascent step.

        Arguments:
            l {float} -- temporal bandwidth.

        Returns:
            float -- new temporal bandwidth.
        """

        l_torch = torch.tensor(l, dtype=torch.float64, requires_grad = True)
        
        loglikelihood = compute_loglikelihood(l_torch)
    
        loglikelihood.backward(l_torch)
        grad = l_torch.grad
        grad = float(grad)
        
        l += learning_rate * grad
        epsilon = 0.000001
        return np.clip(l, 1/samples + epsilon, 1)
    
    def gradient_ascent_loop(l, max_iter=50, min_diff=0.01):
        """ Performs a complete loop of gradient ascent

        Arguments:
            l {float} -- temporal bandwidth.

        Keyword Arguments:
            max_iter {int} -- maximum number of iterations (default: {50})
            min_diff {float} -- minimum differecence between iterations below which the gradient loop is terminated (default: {0.01})

        Returns:
            float -- new temporal bandwidth.
        """

        i = 0
        while True:
            i += 1
            old_l = l
            l = gradient_ascent_step(l)
            #print("I:", i, "lambda:", l)
            if max_iter is not None and i > max_iter:
                break
            elif epsilon is not None and np.abs(l - old_l) < min_diff:
                break
        
        return l

    epsilon = 0.000001
    points = 20
    
    if optimization == "gradient-ascent":   
        ls = np.linspace(1/samples + epsilon, 1, num = points)
        ls = [gradient_ascent_loop(l) for l in ls]
        ls = [torch.tensor(l) for l in ls]
        lks = [compute_loglikelihood(l).numpy() for l in ls]
        #for l, lk in zip(ls, lks):
        #    print("Lambda: {0:4.3f}, Log-Likelihood: {1:8.6f}".format(l, lk))  
        l = ls[np.nanargmax(lks)]
    elif optimization == "grid-search":
        ls = np.linspace(1/samples + epsilon, 1, num = points)
        lks = [compute_loglikelihood(l).numpy() for l in ls]
        #for l, lk in zip(ls, lks):
        #    print("Lambda: {0:4.3f}, Log-Likelihood: {1:8.6f}".format(l, lk))   
        l = ls[np.nanargmax(lks)]
    else:
        print("Invalid optimization method:", optimization)
    
    print("Selected Lambda: {:6.3f}".format(l))
    
    return temporal_kernel(samples, l, kernel)

presets = {
    "likelihood": likelihood_kernel_weights
}

def temporal_weights_calculator(weights, samples, preset="fixed", _lambda=1, h=1, kernel="epanechnikov"):
    
    if preset == "fixed":
        return temporal_kernel(samples, _lambda, kernel)
    
    if preset not in presets:
        print("Preset not found.")
        return None
    
    return presets[preset](weights, samples, h, kernel=kernel)
