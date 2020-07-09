import numpy as np

def tnorm(a,b,name):
    if name is 'min':
        return tnorm_min(a,b)
    elif name is 'prod':
        return tnorm_product(a,b)

def tnorm_product(values):
    return np.prod(values,axis=0)

def tnorm_minimum(values):
    return values.min(axis=0)
