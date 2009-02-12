import numpy as npy

def safe_sqr(u_j):
    return npy.where(abs(u_j) < 1e-160, 0, u_j)**2
