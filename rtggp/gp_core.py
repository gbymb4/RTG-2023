# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:08:22 2023

@author: Gavin
"""

import operator

import numpy as np

from deap import gp, creator, base, tools

from .gp_types import (
    Vector1D,
    Vector2D,
    IntSeed,
    IntConst,
    IntWindow,
    FloatConst,
    NumericUnion,    
)

from .gp_funcs import (
    identity,
    sum_last_axis,
    avg_last_axis,
    max_last_axis,
    min_last_axis,
    dot_last_axis,
    #sqrt, ln,
    add, sub,
    mul, div,
    gt, lt,
    harmonic_weighting,
    random_weighting,
    moving_average,
    rate_of_change,
    sigmoid,
    relu,
    sample_bids,
    sample_asks,
    sample_bids_volumes,
    sample_asks_volumes,
    sample_mid_prices
)

def init_dcstgp():...
    
def init_scstgp(**kwargs):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    pset = gp.PrimitiveSetTyped('MAIN', [Vector2D, Vector2D, Vector2D, Vector2D, Vector1D], NumericUnion)

    toolbox = base.Toolbox()

    base_setup(pset, toolbox, **kwargs)
    
    return pset, toolbox
    
def init_dcmtgp():...

def base_setup(pset, toolbox, max_height=16):
    pset.addPrimitive(identity, [IntConst], NumericUnion)
    pset.addPrimitive(identity, [FloatConst], NumericUnion)
    pset.addPrimitive(identity, [IntSeed], IntSeed)
    pset.addPrimitive(identity, [IntWindow], IntWindow)
    
    pset.addPrimitive(sum_last_axis, [Vector1D], NumericUnion)
    pset.addPrimitive(sum_last_axis, [Vector2D], Vector1D)
    
    pset.addPrimitive(avg_last_axis, [Vector1D], NumericUnion)
    pset.addPrimitive(avg_last_axis, [Vector2D], Vector1D)
    
    pset.addPrimitive(max_last_axis, [Vector1D], NumericUnion)
    pset.addPrimitive(max_last_axis, [Vector2D], Vector1D)
    
    pset.addPrimitive(min_last_axis, [Vector1D], NumericUnion)
    pset.addPrimitive(min_last_axis, [Vector2D], Vector1D)
    
    pset.addPrimitive(dot_last_axis, [Vector1D, Vector1D], NumericUnion)
    pset.addPrimitive(dot_last_axis, [Vector2D, Vector2D], Vector1D)
    
    pset.addPrimitive(add, [NumericUnion, Vector1D], Vector1D)
    pset.addPrimitive(add, [NumericUnion, Vector2D], Vector2D)
    pset.addPrimitive(add, [Vector1D, Vector1D], Vector1D)
    pset.addPrimitive(add, [Vector2D, Vector2D], Vector2D)
    
    pset.addPrimitive(sub, [NumericUnion, Vector1D], Vector1D)
    pset.addPrimitive(sub, [NumericUnion, Vector2D], Vector2D)
    pset.addPrimitive(sub, [Vector1D, Vector1D], Vector1D)
    pset.addPrimitive(sub, [Vector2D, Vector2D], Vector2D)
    
    pset.addPrimitive(mul, [NumericUnion, Vector1D], Vector1D)
    pset.addPrimitive(mul, [NumericUnion, Vector2D], Vector2D)
    pset.addPrimitive(mul, [Vector1D, Vector1D], Vector1D)
    pset.addPrimitive(mul, [Vector2D, Vector2D], Vector2D)
    
    pset.addPrimitive(div, [NumericUnion, Vector1D], Vector1D)
    pset.addPrimitive(div, [NumericUnion, Vector2D], Vector2D)
    pset.addPrimitive(div, [Vector1D, Vector1D], Vector1D)
    pset.addPrimitive(div, [Vector2D, Vector2D], Vector2D)
    
    pset.addPrimitive(gt, [NumericUnion, Vector1D], Vector1D)
    pset.addPrimitive(gt, [NumericUnion, Vector2D], Vector2D)
    pset.addPrimitive(gt, [Vector1D, Vector1D], Vector1D)
    pset.addPrimitive(gt, [Vector2D, Vector2D], Vector2D)
    
    pset.addPrimitive(lt, [NumericUnion, Vector1D], Vector1D)
    pset.addPrimitive(lt, [NumericUnion, Vector2D], Vector2D)
    pset.addPrimitive(lt, [Vector1D, Vector1D], Vector1D)
    pset.addPrimitive(lt, [Vector2D, Vector2D], Vector2D)
    
    pset.addPrimitive(harmonic_weighting, [Vector1D], Vector1D)
    pset.addPrimitive(harmonic_weighting, [Vector2D], Vector2D)
    
    pset.addPrimitive(random_weighting, [Vector1D, IntSeed], Vector1D)
    pset.addPrimitive(random_weighting, [Vector2D, IntSeed], Vector2D)
    
    pset.addPrimitive(moving_average, [Vector1D, IntWindow], Vector1D)
    
    pset.addPrimitive(rate_of_change, [Vector1D], Vector1D)
    
    pset.addPrimitive(sigmoid, [NumericUnion], NumericUnion)
    pset.addPrimitive(sigmoid, [Vector1D], Vector1D)
    pset.addPrimitive(sigmoid, [Vector2D], Vector2D)
    
    pset.addPrimitive(relu, [NumericUnion], NumericUnion)
    pset.addPrimitive(relu, [Vector1D], Vector1D)
    pset.addPrimitive(relu, [Vector2D], Vector2D)
    
    pset.addEphemeralConstant('const_float', lambda: np.random.rand(1)[0] * 10, NumericUnion)
    pset.addEphemeralConstant('const_int', lambda: np.random.randint(1, 11), IntConst)
    pset.addEphemeralConstant('window', lambda: np.random.randint(1, 21), IntWindow)
    pset.addEphemeralConstant('seed', lambda: np.random.randint(0, 2 ** 31), IntSeed)

    toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register('mate', gp.cxOnePoint)
    toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value=max_height)) 
    toolbox.register('mutate', gp.mutNodeReplacement, pset=pset)
    toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value=max_height))
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('compile', gp.compile)

class SCSTGPSignalStrategy:
    
    def __init__(self, func, scope):
        self.func = func
        self.scope = scope
        
    def __call__(self, history):
        etf_asks = sample_asks(history, self.scope, False)
        etf_bids = sample_bids(history, self.scope, False)
        
        etf_asks_volumes = sample_asks_volumes(history, self.scope, False)
        etf_bids_volumes = sample_bids_volumes(history, self.scope, False)
        
        etf_mid_prices = sample_mid_prices(history, self.scope, False)
        
        hedge_asks = sample_asks(history, self.scope, True)
        hedge_bids = sample_bids(history, self.scope, True)
        
        hedge_asks_volumes = sample_asks_volumes(history, self.scope, True)
        hedge_bids_volumes = sample_bids_volumes(history, self.scope, True)
        
        hedge_mid_prices = sample_mid_prices(history, self.scope, True)
        
        etf = self.func(
            etf_asks,
            etf_bids,
            etf_asks_volumes,
            etf_bids_volumes,
            etf_mid_prices
        )
        
        hedge = self.func(
            hedge_asks,
            hedge_bids,
            hedge_asks_volumes,
            hedge_bids_volumes,
            hedge_mid_prices
        )
        
        return etf - hedge