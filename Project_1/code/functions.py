"""
Various functions that are used throughout
"""
import numpy as np


def laplace(pos, alpha):
    laplace = 0
    step = 0.05
    for i in range(len(pos)): #iterate for each particle
        for j in range(len(pos[0,:])): #iterate for each dimension
            
            pos[i,j] += step
            forward = repeat(pos, alpha)
            pos[i,j] -= 2*step
            backward = repeat(pos, alpha)
            pos[i,j] += step
            middle = repeat(pos, alpha)
            
            
            laplace += (forward + backward - 2*middle)/(middle*step**2)
            
    return laplace

def repeat(pos, alpha, re=False):
    res = 0
    for i in range(len(pos)): #iterate for each particle
        r2 = 0
        for j in range(len(pos[0,:])): #iterate for each dimension
            r2 += pos[i,j]**2
        res += r2
       
        
    if re == True:
        return -res*np.exp(-alpha*res)
    else:
        return np.exp(-alpha*res)


def gradient(pos, alpha):
    gradient = np.zeros((len(pos), len(pos[0,:])))
    for i in range(len(gradient)):
        for j in range(len(gradient[0,:])):
            gradient[i,j] = -2*alpha*pos[i,j]
        