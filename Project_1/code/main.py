"""
Below are a few select example runs for each of the different parts of the project
"""
import numpy as np
from tools.runner import runner, runParallel
from tools.data import Data
from classes.MonteCarlo import MonteCarlo
from models.nonInteractive import nonInteractive
from classes.System import System
import time
import matplotlib.pyplot as plt



def part_b():
    #alphas = np.linspace(0.3,0.7, 5)
    t0 =  time.perf_counter()
    store = []
    store2 = []
    store3 = []
#    for alpha in alphas:
    for i in range(1):
        alpha = 0.5
        res = runner(1000, 10, 3, alpha, 0.01, impo=True, ana=False, write_to=False) 
#        res2 = runner(1000, 10, 3, alpha, 0.01, impo=True, ana=False, write_to=False) 
#        res3 = runner(1000, 25, 3, alpha, 0.01, impo=False, ana=False, write_to=False) 
        store.append(res[:,1].sum()/(1000))
#        store2.append(res2[:,1].sum()/(1000))
#        store3.append(res3[:,1].sum()/(1000*25))
        for i in range(len(res[0,:])):    
            
            print('Brute:', res[:,i].sum()/1000, '\n')
#            print('Impo:', res2[:,i].sum()/1000, '\n')
    t1 = time.perf_counter() - t0
    print('Total runtime for %g different alpha values and %g iterations: %gs' %(1, itrs, t1))
    input("Press Enter to continue...")
    
    
#    plt.plot(alphas, store)
#    plt.plot(alphas, store2)
#    plt.plot(alphas, store3)
#    plt.legend(['1 particles', '10 particles', '25 particles'])
#    plt.title('Energies for different numbers of particles normalized.')
#    plt.xlabel('$a$')
#    plt.ylabel('$E_L(a)/N$')
#    plt.show()
    
def part_c():
    t0 =  time.perf_counter()
    dts = [10, 5, 1, 0.1, 0.01, 0.001, 0.01]
    store = []
    for dt in dts:
        res = runner(1000, 10, 3, 0.5, dt, impo=True, ana=False, write_to=False)  
        
        store.append(res[:,1].sum()/1000)
        for i in range(len(res[0,:])):    
            print(res[:,i].sum()/1000, '\n')
    t1 = time.perf_counter() - t0
    print('Total runtime for %g different dt values and %g iterations: %gs' %(len(dts), itrs, t1))
    input("Press Enter to continue...")
    
    plt.plot(np.log(dts), store)
    plt.xlabel('$\log{\delta t}$')
    plt.ylabel('$E_L(\delta t)$')
    plt.title('Energy as a function of $\delta$ t')
    plt.show()
    
def part_d():
    t0 =  time.perf_counter()
    alpha = 0.6
    dt = 0.05
    learn = 0.3
    
    model = nonInteractive(100, 3, alpha)
    
    system = System(model)
    
    mc = MonteCarlo(system)
    
    alphasList, varsList = mc.gradientDescent(10000, dt, learn, True)
    
    t1 = time.perf_counter() - t0
    print('Optimal Alpha value %g found after %gs' %(alphasList[-1], t1))
    f = open('../data/GD_a%g_dt%g_eta%g.txt' %(alpha, dt, learn), 'w')
    for i in range(len(alphasList)):
        f.write('%g, %g\n' %(alphasList[i], varsList[i]))
    f.close()
    input("Press Enter to continue...")
    
def part_d2():
    dat = Data('../data/GD_a0.4_dt0.05_eta0.25.txt'); dat.fileToArray(); dat.plotData()
    dat = Data('../data/GD_a0.6_dt0.1_eta0.25.txt'); dat.fileToArray(); dat.plotData()
    dat = Data('../data/GD_a0.6_dt0.05_eta0.3.txt'); dat.fileToArray(); dat.plotData()
    dat = Data('../data/GD_a0.6_dt0.05_eta0.25.txt'); dat.fileToArray(); dat.plotData()
    plt.xlabel('Iterations')
    plt.ylabel('$a$ or variance')
    plt.title('$a$ and variance evolution over iterations')
    plt.legend(['a: 0.4, dt: 0.05, $\eta$: 0.25','a: 0.4, dt: 0.05, $\eta$: 0.25',
                'a: 0.6, dt: 0.1, $\eta$: 0.25', 'a: 0.6, dt: 0.1, $\eta$: 0.25',
                'a: 0.6, dt: 0.05, $\eta$: 0.3', 'a: 0.6, dt: 0.05, $\eta$: 0.3',
                'a: 0.6, dt: 0.05, $\eta$: 0.25', 'a: 0.6, dt: 0.05, $\eta$: 0.25'])
    plt.show()

def part_e():
    dat = Data('../data/run_1226_01-04.txt')

    dat.fileToArray()
    
    
    dat.bootStrap(250)
    
    
    input("Press Enter to continue...")
    
def part_f():
    threads=4
    t0 =  time.perf_counter()
    itrs, res = runParallel(10000,100,3,0.5,0.01,threads,write_to=False) #is technically parallelized but is slower. copy.deepcopy = culprit?!
    
    for i in range(len(res[0,:])):    
            print(res[:,i].sum()/itrs, '\n')
    t1 = time.perf_counter() - t0
    print('Total runtime with %g threads and %g iterations: %gs' %(threads, itrs, t1))
    input("Press Enter to continue...")

def part_g():
    ...

def part_h():
    ...
    
itrs = 10000




if __name__ == '__main__': #must have for parallelization to work (on windows)
    
    part_b()
#    part_c()
#    part_d()
#    part_d2()
#    part_e()
#    part_f()