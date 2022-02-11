import numpy as np


def logit2(beta):
    return np.log2(beta / (1.0 - beta))


def expit2(mval):
    return np.power(2.0, mval) / (np.power(2.0, mval) + 1.0)
