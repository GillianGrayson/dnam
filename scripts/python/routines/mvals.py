import numpy as np


def logit2(beta, alpha):
    return np.log2((beta + alpha)/ (1.0 - beta + alpha))


def expit2(mval):
    return np.power(2.0, mval) / (np.power(2.0, mval) + 1.0)
