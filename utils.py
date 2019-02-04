import math
import numpy as np


def sinusoid(x):
    assert x.shape == (2, )
    return math.sin(10 * x[0]) + math.cos(4 * x[1]) - math.cos(3 * x[0] * x[1])


def zero_mean(x):
    return 0


class squared_exponential_kernel(object):
    def __init__(self, sigma, l):
        self.sigma = sigma
        self.l = l

    def __call__(self, x1, x2):
        assert x1.shape == x2.shape
        if x1.shape == (1, ):
            return np.square(self.sigma) * np.exp(-np.square(x1 - x2) /
                                                  (2 * np.square(self.l)))
        else:
            return np.square(self.sigma) * np.exp(
                -np.square(np.linalg.norm(x1 - x2, ord=2)) /
                (2 * np.square(self.l)))

    def update(sigma):
        self.sigma = sigma