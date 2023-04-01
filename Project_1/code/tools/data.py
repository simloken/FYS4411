import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import matplotlib.pyplot as plt

class Data:
    
    def __init__(self, file):
        self.file = file
        
        
    def fileToArray(self):
        f = open(self.file, 'r').readlines()
        lst = []   
        for i in f:
            lst.append(i.strip('\n').split(','))
        
        self.dataArray = np.array(lst).astype(float)
        self.dataArray = self.dataArray[:,1:]
        self.dataArray = self.dataArray[:,1]
        self.dataArray = np.split(self.dataArray, 5)
        
    def bootStrap(self, N):
        
        boArr = np.zeros(N)
        for j in range(len(self.dataArray)):
            for i in range(N):
                boArr[i] = np.average(np.random.choice(self.dataArray[j][:], len(self.dataArray[j])))
            print('Variance: ', np.var(boArr))
            print('Error: ', np.std(boArr))
            
            
    def blocking(self):
        ... #???
        
        
        
    def plotData(self): #plot alpha convergence for GD
        f = open(self.file, 'r').readlines()
        lst = []
        lst2 = []
        for i in f:
            i = i.split(',')
            lst.append(i[0])
            lst2.append(i[1])
            
        lst = np.array(lst).astype(float)
        lst2 = np.array(lst2).astype(float)
            
            
        plt.plot(range(len(lst)), lst)
        plt.plot(range(len(lst)), lst2, alpha=0.8, linestyle='--')