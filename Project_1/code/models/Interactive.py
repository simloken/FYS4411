"""
Class which handles energies and drift for an interactive model.
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from functions import laplace

import numpy as np

class nonInteractive:
    
    def __init__(self, parts, dims, alpha, a):
        self.parts = np.zeros((parts, dims))
        self.dims = dims
        self.alpha = alpha
        self.gamma
        self.a = a
        
    def Energy(self, pos, alpha, ana=False):
        
        if ana: #analytical
            r2 = 0
            for i in range(len(pos)): #iterate for each particle
                for j in range(len(pos[0,:])): #iterate for each dimension
                    r2 += pos[i,j]**2
            
            return len(pos)*len(pos[0,:])*alpha + (1-4*alpha**2)*r2/2
        
        else: #numerical
            PE = 0
            for i in range(len(pos)): #iterate for each particle
                r2 = 0
                for j in range(len(pos[0,:])): #iterate for each dimension
                    if j != 2:
                        r2 += pos[i,j]**2
                    else:
                        r2 += pos[i,j]**2*self.gamma**2
                PE += r2
                
                for j in range(len(pos[0,:])):
                    if np.linalg.norm(pos[i,:] - pos[j,:]) <= self.a:
                        PE += 1e12 #some arbitrary high number to represent infinity as the particle comes closer than a
            
            KE = -laplace(pos, alpha)/2 #kinetic energy remains the same
            
            return KE + PE/2
    
    def Drift(self, pos, alpha):
        gradient = np.zeros((len(pos), len(pos[0,:])))
        for i in range(len(gradient)):
            for j in range(len(gradient[0,:])):
                gradient[i,j] = -2*alpha*pos[i,j]
                
                
        return 2*gradient