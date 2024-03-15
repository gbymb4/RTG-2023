# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:09:32 2023

@author: Gavin
"""

import functools

import numpy as np

from .gp_core import SCSTGPSignalStrategy

from deap import algorithms, tools, creator
from simulation import Simulator

pset = None

def exec_scstgp_optim(
        pset,
        toolbox, 
        scope=50,
        pop_size=128, 
        max_height=16, 
        signal_threshold=50, 
        **kwargs
    ):
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register('avg', np.mean)
    stats.register('min', np.min)
    stats.register('max', np.max)
    stats.register('std', np.std)
    
    toolbox.register('evaluate', functools.\
                     partial(
                         eval_scstgp, 
                         pset=pset, 
                         toolbox=toolbox, 
                         scope=scope,
                         signal_threshold=signal_threshold
                    )
                )
    
    pop, log = algorithms.eaSimple(
        pop, 
        toolbox,
        stats=stats, 
        halloffame=hof,
        **kwargs
    )
    
    best = hof[0]
    
    return best, log

def eval_scstgp(indi, pset, toolbox, scope, signal_threshold):
    func = toolbox.compile(indi, pset)  
    
    sig = SCSTGPSignalStrategy(func, scope)
    sim = Simulator(sig, signal_threshold=signal_threshold)
            
    sim.run()
        
    return sim.instantaneous_profit(),