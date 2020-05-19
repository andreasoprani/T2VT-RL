import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy, norm
from scipy.integrate import nquad
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
        if x < 0 or x > 1:
            return 0
        else:
            return (3/4) * (1 - x**2)
    
    def full_estimator(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, 
                       x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, 
                       x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, 
                       x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, 
                       x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, 
                       x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, 
                       x120, x121, x122, x123, x124, x125, x126, x127, x128, x129, x130, x131, x132, x133, x134, x135, x136, x137, x138, x139, 
                       x140, x141, x142, x143, x144, x145, x146, x147, x148, x149, x150, x151, x152, x153, x154, x155, x156, x157, x158, x159, 
                       x160, x161, x162, x163, x164, x165, x166, x167, x168, x169, x170, x171, x172, x173, x174, x175, x176, x177, x178, x179, 
                       x180, x181, x182, x183, x184, x185, x186, x187, x188, x189, x190, x191, x192, x193, x194, x195, x196, x197, x198, x199,
                       x200, x201, x202, x203, x204, x205, x206, x207, x208, x209, x210, x211, x212, x213, x214, x215, x216, x217, x218, x219, 
                       x220, x221, x222, x223, x224, x225, x226, x227, x228, x229, x230, x231, x232, x233, x234, x235, x236, x237, x238, x239, 
                       x240, x241, x242, x243, x244, x245, x246, x247, x248, x249, x250, x251, x252, x253, x254, x255, x256, x257, x258, x259, 
                       x260, x261, x262, x263, x264, x265, x266, x267, x268, x269, x270, x271, x272, x273, x274, x275, x276, x277, x278, x279, 
                       x280, x281, x282, x283, x284, x285, x286, x287, x288, x289, x290, x291, x292, x293, x294, x295, x296, x297, x298, x299, 
                       x300, x301, x302, x303, x304, x305, x306, x307, x308, x309, x310, x311, x312, x313, x314, x315, x316, x317, x318, x319, 
                       x320, x321, x322, x323, x324, x325, x326, x327, x328, x329, x330, x331, x332, x333, x334, x335, x336, x337, x338, x339, 
                       x340, x341, x342, x343, x344, x345, x346, x347, x348, x349, x350, x351, x352, x353, x354, x355, x356, x357, x358, x359, 
                       x360, x361, x362, x363, x364, x365, x366, x367, x368, x369, x370, x371, x372, x373, x374, x375, x376, x377, x378, x379, 
                       x380, x381, x382, x383, x384, x385, x386, x387, x388, x389, x390, x391, x392, x393, x394, x395, x396, x397, x398, x399, 
                       x400, x401, x402, x403, x404, x405, x406, x407, x408, x409, x410, x411, x412, x413, x414, x415, x416, x417, x418, x419, 
                       x420, x421, x422, x423, x424, x425, x426, x427, x428, x429, x430, x431, x432, x433, x434, x435, x436, x437, x438, x439, 
                       x440, x441, x442, x443, x444, x445, x446, x447, x448, x449, x450, x451, x452, x453, x454, x455, x456, x457, x458, x459, 
                       x460, x461, x462, x463, x464, x465, x466, x467, x468, x469, x470, x471, x472, x473, x474, x475, x476, x477, x478, x479, 
                       x480, x481, x482, x483, x484, t0):
        """Full estimator function decomposed

        Arguments:
            x0 {float} -- t
            x1 -> x484 {float} -- Q elements
            t0 {float} -- lambda
        """
        
        temporal_components = np.zeros(samples, dtype=np.float128)
        spatial_components = np.zeros(samples, dtype=np.float128)
        
        q = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, 
            x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31, x32, x33, x34, x35, x36, x37, x38, x39, 
            x40, x41, x42, x43, x44, x45, x46, x47, x48, x49, x50, x51, x52, x53, x54, x55, x56, x57, x58, x59, 
            x60, x61, x62, x63, x64, x65, x66, x67, x68, x69, x70, x71, x72, x73, x74, x75, x76, x77, x78, x79, 
            x80, x81, x82, x83, x84, x85, x86, x87, x88, x89, x90, x91, x92, x93, x94, x95, x96, x97, x98, x99, 
            x100, x101, x102, x103, x104, x105, x106, x107, x108, x109, x110, x111, x112, x113, x114, x115, x116, x117, x118, x119, 
            x120, x121, x122, x123, x124, x125, x126, x127, x128, x129, x130, x131, x132, x133, x134, x135, x136, x137, x138, x139, 
            x140, x141, x142, x143, x144, x145, x146, x147, x148, x149, x150, x151, x152, x153, x154, x155, x156, x157, x158, x159, 
            x160, x161, x162, x163, x164, x165, x166, x167, x168, x169, x170, x171, x172, x173, x174, x175, x176, x177, x178, x179, 
            x180, x181, x182, x183, x184, x185, x186, x187, x188, x189, x190, x191, x192, x193, x194, x195, x196, x197, x198, x199,
            x200, x201, x202, x203, x204, x205, x206, x207, x208, x209, x210, x211, x212, x213, x214, x215, x216, x217, x218, x219, 
            x220, x221, x222, x223, x224, x225, x226, x227, x228, x229, x230, x231, x232, x233, x234, x235, x236, x237, x238, x239, 
            x240, x241, x242, x243, x244, x245, x246, x247, x248, x249, x250, x251, x252, x253, x254, x255, x256, x257, x258, x259, 
            x260, x261, x262, x263, x264, x265, x266, x267, x268, x269, x270, x271, x272, x273, x274, x275, x276, x277, x278, x279, 
            x280, x281, x282, x283, x284, x285, x286, x287, x288, x289, x290, x291, x292, x293, x294, x295, x296, x297, x298, x299, 
            x300, x301, x302, x303, x304, x305, x306, x307, x308, x309, x310, x311, x312, x313, x314, x315, x316, x317, x318, x319, 
            x320, x321, x322, x323, x324, x325, x326, x327, x328, x329, x330, x331, x332, x333, x334, x335, x336, x337, x338, x339, 
            x340, x341, x342, x343, x344, x345, x346, x347, x348, x349, x350, x351, x352, x353, x354, x355, x356, x357, x358, x359, 
            x360, x361, x362, x363, x364, x365, x366, x367, x368, x369, x370, x371, x372, x373, x374, x375, x376, x377, x378, x379, 
            x380, x381, x382, x383, x384, x385, x386, x387, x388, x389, x390, x391, x392, x393, x394, x395, x396, x397, x398, x399, 
            x400, x401, x402, x403, x404, x405, x406, x407, x408, x409, x410, x411, x412, x413, x414, x415, x416, x417, x418, x419, 
            x420, x421, x422, x423, x424, x425, x426, x427, x428, x429, x430, x431, x432, x433, x434, x435, x436, x437, x438, x439, 
            x440, x441, x442, x443, x444, x445, x446, x447, x448, x449, x450, x451, x452, x453, x454, x455, x456, x457, x458, x459, 
            x460, x461, x462, x463, x464, x465, x466, x467, x468, x469, x470, x471, x472, x473, x474, x475, x476, x477, x478, x479, 
            x480, x481, x482, x483, x484]
        
        l = t0
        
        for j in range(samples):
            
            q_j = np.array(weights[j], dtype=np.float128)
            
            temporal_components[j] = epanechnikov(np.abs(x0-j/samples)/l)
            spatial_components[j] = norm.pdf(np.linalg.norm(q - q_j)/h)
            
        temporal_components = [t / np.sum(temporal_components) for t in temporal_components]
        
        dim = len(q)
        
        print(np.dot(temporal_components, spatial_components) / (samples * l * (h**dim)))
        
        return np.dot(temporal_components, spatial_components) / (samples * l * (h**dim)) 
    
    def estimator_without_i(i, l):
        
        t_i = i/samples
        q_i = np.array(weights[i], dtype=np.float128)
        
        temporal_components = np.zeros(samples - 1, dtype=np.float128)
        spatial_components = np.zeros(samples - 1, dtype=np.float128)
        
        index = 0
        
        for j in range(samples):
            
            if j == i:
                continue
            
            q_j = np.array(weights[j], dtype=np.float128)
            
            temporal_components[index] = epanechnikov(np.abs(i-j)/(samples*l))
            spatial_components[index] = norm.pdf(np.linalg.norm(q_i - q_j)/h)
            
            print("J:", j)
            print("Kt:", temporal_components[index])
            print("Ks:", spatial_components[index])
            
            index += 1
            
        temporal_components = [t / np.sum(temporal_components) for t in temporal_components]
        
        dim = len(q_i)
        print(dim)
        
        return np.dot(temporal_components, spatial_components) / (samples * l * (h**dim)) 
    
    def calculate_cv(l):
        
        dim = len(weights[0])
        
        # TODO integral calculation
        ranges = [[0,1]]
        for i in range(dim):
            ranges.append([-np.inf, np.inf])
        integral_component, err = nquad(full_estimator, ranges, args = [l], opts = {'epsrel': 1})
        
        print("Integral:", integral_component)
        print("Error:", err)
    
        estimator_component = (2 / samples) * np.sum([estimator_without_i(i, l)])
        
        print("Estimator component:", estimator_component)
        
        return integral_component - estimator_component
      
    calculate_cv(0.5)
        
    return True

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
