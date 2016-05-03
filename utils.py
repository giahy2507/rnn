__author__ = 'HyNguyen'

import numpy as np



def init_w(rng, shape):
    return rng.randn(*shape) * np.sqrt(1 / shape[0]+1)

def init_b(shape):
    return np.zeros(shape)

