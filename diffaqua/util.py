from itertools import product

import numpy as np


def evenly_dist_weights(num_weights, dim):
    rets = [ret for ret in product(np.linspace(0.0, 1.0, num_weights), repeat=dim) if round(sum(ret), 3) == 1.0]
    return rets
