"""
A universal "backend" that handles inputs and and inputs them into classes to do numerical calculations.
Contains both a non-parallelized and a parallelized function, although the parallelized one is somehow slower (?)
"""
import numpy as np
import multiprocessing as mp

from classes.System import System
from classes.MonteCarlo import MonteCarlo
from models.nonInteractive import nonInteractive



def runParallel(itrs,parts,dims,alpha,dt,impo=False,threads=None,write_to=False):
    if threads==None:
        threads = int(mp.cpu_count())
        
    pool=mp.Pool(int(threads))
    
    if int(itrs) % int(threads) == 0:
        itrSize = int(itrs/threads)
    else:
        itrSize = int(np.floor(itrs/threads))
        
    lst = [itrSize]*threads
    
    results = [pool.apply_async(runner, args=(i, parts, dims, alpha, dt, impo)) for i in lst]
    
    final = []
    for i in results:
        final.extend(i.get())

    pool.close()
    
    
    return itrSize*threads, np.array(final)



def runner(itrs,parts,dims,alpha,dt, impo=False, ana=False, write_to=False):
    
    model = nonInteractive(parts, dims, alpha)
    
    system = System(model, ana)
    
    mc = MonteCarlo(system, write_to)
    
    results = mc.VMC(itrs, 0.0001, alpha, impo)
    
    #mc.gradientDescent(10000, 0.0001, 0.5, 0.1, True)
    
    return results