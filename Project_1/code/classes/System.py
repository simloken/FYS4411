"""
Class for finding general information about the system in its current state
"""
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from functions import repeat

import numpy as np

class System:
    
    def __init__(self, model, ana=False):
        self.model = model #model object
        self.parts = self.model.parts
        self.dims = self.model.dims
        self.ana = ana
        
        self.E_L = 0 #local energy
        self.driftForce = np.zeros((len(self.parts), self.dims))
        
        self.wf = 0 #value of wave function
        self.dwf = 0 #value of derivative of wave function
        
        for i in range(len(self.parts)): #initialize random positions
            for j in range(self.dims):
                self.parts[i][j] = np.random.rand(1)-0.5
    
    def findWaveFunction(self, alpha):
        res = 0
        for i in range(len(self.parts)): #iterate for each particle
            r2 = 0
            for j in range(len(self.parts[0,:])): #iterate for each dimension
                r2 += self.parts[i,j]**2
                
        res += r2

        self.wf = np.exp(-alpha*res)
    
    def findEnergy(self, alpha):
        self.E_L = self.model.Energy(self.parts, alpha, self.ana)
        
        
        
    def findDrift(self, alpha):
        self.driftForce = self.model.Drift(self.parts, alpha)
        
        
    def findDerivative(self):
        self.derivative = repeat(self.parts, re=True)
            
    def variance(self):
        self.variance = self.E_L2 - self.E_L**2
        
    def store(self):
        
        self.storedParts = self.parts
        self.storedDims = self.dims
        self.storedE_L = self.E_L
        self.storeddriftForce = self.driftForce
        self.storedwf = self.wf
        self.storeddwf = self.dwf
        
        
    def retrieve(self):
        self.parts = self.storedParts
        self.dims = self.storedDims
        self.E_L = self.storedE_L
        self.driftForce = self.storeddriftForce
        self.wf = self.storedwf
        self.dwf = self.storeddwf
        
        
        