import numpy as np
import copy
import os
import sys
import inspect
import time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from functions import repeat

class MonteCarlo:
    def __init__(
            self,
            system,
            write=False):
        self.system = system
        self.write = write
        
    def VMC(self, itrs, dt, alpha, impo=False): #returns array of shape [itrs, 3]
        
        results = []
        variance = []
        
        for i in range(itrs):
            itr_system = copy.deepcopy(self.system); movedSystem = copy.deepcopy(itr_system) #this is slow. fix?
            itr_system.findWaveFunction(alpha)
            itr_system.findDrift(alpha)
            
            accepted = 0
            E = 0
            E2 = 0
            prevE = 0
            
            for j in range(len(itr_system.parts)):
                
                movedSystem.parts[j] = self.Move(movedSystem.parts[j], movedSystem, movedSystem.driftForce[j], dt, alpha, impo)
                    
                
                
                movedSystem.findEnergy(alpha)
                
                E += (movedSystem.E_L - prevE)# sum of energy deltas
                E2 += (movedSystem.E_L - prevE)**2 #sum of squared energy deltas
                prevE = movedSystem.E_L

                movedSystem.findWaveFunction(alpha)
                
                P = movedSystem.wf**2/itr_system.wf**2
                                
                if impo: #green's function
                    movedSystem.findDrift(alpha)
                    g = 0
                    for k in range(len(itr_system.parts[j])):
                        g += ((itr_system.parts[j,k] - movedSystem.parts[j,k])*(itr_system.driftForce[j,k] + 
                              movedSystem.driftForce[j,k]) + (0.5*dt)*(itr_system.driftForce[j,k]**2 - 
                            movedSystem.driftForce[j,k]**2))
                
                    P *= np.exp(0.5*g)
                
                if np.random.rand(1) <= P: #test
                    accepted += 1
                    itr_system.parts[j] = movedSystem.parts[j]
                    
                else:
                    movedSystem.parts[j] = itr_system.parts[j]
                
            
            variance.append(E2-E**2)
            
            itr_system.findWaveFunction(alpha)
            itr_system.findEnergy(alpha)
            
            results.append([itr_system.wf, #storage of results after a cycle
                            itr_system.E_L, accepted/len(itr_system.parts)])
        
    
        variance = np.array(variance).sum()/len(itr_system.parts) #why is this zero??
        
        err = np.sqrt(abs(variance))
        print('For alpha: %g' %alpha)
        print('Variance: %g\nError: %g\n' %(variance, err))
        
        resultsArray = np.array(results) #retrieve by resultsArray[:, x] where x in [0,2], wf: 0, E_L: 1, acc.rate: 2
        
        if self.write:
            from datetime import date
            f = open('../data/run_%s_%s.txt' %(time.strftime('%H%M', time.localtime()), date.today().strftime("%d-%m")), 'a')
            for x in resultsArray:
                f.write('%g, %g, %g, %g\n' %(alpha, x[0], x[1], x[2]))
                
            f.close()
            
            f = open('../data/avgrun_%s_%s.txt'%(time.strftime('%H%M', time.localtime()), date.today().strftime("%d-%m")), 'a')   
            f.write('%g, %g, %g, %g\n' %(alpha, resultsArray[:,0].sum()/itrs,resultsArray[:,1].sum()/itrs,resultsArray[:,0].sum()/itrs))
            
            f.close()
            
        
        return resultsArray
                
            
                
                
                
    def Move(self, part, system, drift, dt, alpha, impo=False):
        if impo==False:
            for i in range(len(part)):
                part[i] += np.random.rand(1) - 0.5       
        else:
            for i in range(len(part)):
                part[i] += drift[i]*dt + np.sqrt(dt)*np.random.normal()
            
        return part
    
    
   
    def gradientDescent(self, itrs, dt, eta, impo):
        
        tol = 1e-4
        varsList = []
        alphasList = []
        alpha = self.system.model.alpha
        alphasList.append(alpha)
        for i in range(itrs):
            E = 0
            E2 = 0
            prevE = 0
            heldSystem = copy.deepcopy(self.system)
            for j in range(len(heldSystem.parts)):
                heldSystem.parts[j] = self.Move(heldSystem.parts[j], heldSystem, heldSystem.driftForce[j], dt, alpha, impo)
                heldSystem.findEnergy(alpha)
                
                E += (heldSystem.E_L - prevE)# sum of energy deltas
                E2 += (heldSystem.E_L - prevE)**2 #sum of squared energy deltas
                prevE = heldSystem.E_L
                
            
            
            
            variance = (E2-E**2)/itrs
            varsList.append(variance)
            print('VAR:', variance)
            if abs(variance) <= tol:
                print('Alpha value found: %g\nIt took %g iterations' %(alpha, i))
                alphasList.append(alpha)
                varsList.insert(0, varsList[0])
                break
            else:
                alpha -= variance*eta
                alphasList.append(alpha)
                print('ALPHA:', alpha)
            
        return alphasList, varsList


            
            
            
            
        
            