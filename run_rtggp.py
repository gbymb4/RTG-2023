# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:10:23 2023

@author: Gavin
"""

import random, warnings, sys

import numpy as np

from rtggp import init_scstgp, exec_scstgp_optim

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
if __name__ == '__main__':
    seed(int(sys.argv[1]))
    
    warnings.simplefilter('ignore')
    
    pset, toolbox = init_scstgp(max_height=16)
    
    #pool = multiprocessing.Pool()
    #toolbox.register('map', pool.map)

    results = exec_scstgp_optim(
        pset,
        toolbox, 
        cxpb=0.20, 
        mutpb=0.05, 
        ngen=100, 
        verbose=True,
        signal_threshold=10
    )
    
    best, log = results

    print(best)
    print(log)