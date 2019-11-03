import numpy as np

def true_with_probability(p):
    return np.random.choice([True, False], 1, [p, 1 - p])

