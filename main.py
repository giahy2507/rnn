__author__ = 'HyNguyen'
import utils as u

import numpy as np

rng = np.random.RandomState(4488)

a = u.init_w(rng, (300,300))
aa = a[...,None]
print aa.shape
