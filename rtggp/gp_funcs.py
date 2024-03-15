# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:07:27 2023

@author: Gavin
"""

import random

import numpy as np
import pandas as pd

from typing import Union, Dict, Any
from .gp_types import (
    Vector1D,
    Vector2D,
    IntSeed,
    IntWindow,
    IntScope,
    NumericUnion,
    Boolean
)

def identity(x: Any):
    return x

def sum_last_axis(x: Union[Vector1D, Vector2D]):
    return x.sum(axis=len(x.shape)-1)

def avg_last_axis(x: Union[Vector1D, Vector2D]):
    return x.mean(axis=len(x.shape)-1)
    
def max_last_axis(x: Union[Vector1D, Vector2D]):
    return x.max(axis=len(x.shape)-1)

def min_last_axis(x: Union[Vector1D, Vector2D]):
    return x.max(axis=len(x.shape)-1)

def dot_last_axis(
        x: Union[Vector1D, Vector2D],
        y: Union[Vector1D, Vector2D]
    ):
    if len(x.shape) == 2 and len(y.shape) == 2:
        return np.einsum('ij,ij->i', x, y)
    
    return x.dot(y)

def sqrt(x: Union[NumericUnion, Vector1D, Vector2D]):
    return np.sqrt(x)

def ln(x: Union[NumericUnion, Vector1D, Vector2D]):
    return np.log(x)

def add(
        x: Union[NumericUnion, Vector1D, Vector2D],
        y: Union[NumericUnion, Vector1D, Vector2D]
    ):
    return np.add(x, y, casting='unsafe')

def sub(
        x: Union[NumericUnion, Vector1D, Vector2D],
        y: Union[NumericUnion, Vector1D, Vector2D]
    ):
    return np.subtract(x, y, casting='unsafe')

def mul(
        x: Union[NumericUnion, Vector1D, Vector2D],
        y: Union[NumericUnion, Vector1D, Vector2D]
    ):
    return np.multiply(x, y, casting='unsafe')

def div(
        x: Union[NumericUnion, Vector1D, Vector2D],
        y: Union[NumericUnion, Vector1D, Vector2D]
    ):
    ans = x / y
    
    if isinstance(ans, np.ndarray):
        ans[ans == np.nan] = 0
    
    return ans

def gt(
        x: Union[NumericUnion, Vector1D, Vector2D],
        y: Union[NumericUnion, Vector1D, Vector2D]
    ):
    return np.greater(x, y).astype(int)

def lt(
        x: Union[NumericUnion, Vector1D, Vector2D],
        y: Union[NumericUnion, Vector1D, Vector2D]
    ):
    return np.less(x, y).astype(int)

def harmonic_weighting(x: Union[Vector1D, Vector2D]):
    weights = np.array([1 / (i + 1) for i in range(len(x))])
    
    if len(x.shape) == 2:
        weights = weights[:, np.newaxis]
    
    return mul(x, weights[::-1])

def random_weighting(x: Union[Vector1D, Vector2D], seed: IntSeed):
    random.seed(seed)
    np.random.seed(seed)
    
    weights = np.random.rand(len(x))
    
    if len(x.shape) == 2:
        weights = weights[:, np.newaxis]
    
    return mul(x, weights)

def moving_average(x: Vector1D, window: IntWindow) :
    window = int(window)
    
    ma = pd.Series(x).rolling(window).mean().to_numpy()
    
    ma_left = []
    for i in range(1, window):
        vals = x[:i]
        
        if not isinstance(vals, np.ndarray): vals = np.array(vals)
        
        avg = np.mean(vals)
        ma_left.append(avg)

    ma[:len(ma_left)] = ma_left
    return ma

def rate_of_change(x: Vector1D):
    roc = np.diff(x)
    roc = np.concatenate((np.zeros(1), roc))
    
    return roc

def sigmoid(x: Union[NumericUnion, Vector1D, Vector2D]):
    return 1 / (1 + np.exp(-x))

def relu(x: Union[NumericUnion, Vector1D, Vector2D]):
    return mul(x, gt(x, 0))

def sample_bids(
        history: Dict,
        scope: IntScope, 
        is_futures: Boolean
    ):
    prefix = 'etf' if not is_futures else 'hedge'
    
    bids = history[f'{prefix}_bids']
    
    init_idx = max(0, len(bids) - scope)
    vals = np.array(bids[init_idx : len(bids)])
    
    if scope > len(vals):
        padding = np.empty((scope - len(vals), 5))
        padding[:] = np.nan
        
        vals = np.concatenate((padding, vals))
        
    return vals

def sample_asks(
        history: Dict,
        scope: IntScope, 
        is_futures: Boolean
    ):
    prefix = 'etf' if not is_futures else 'hedge'
    
    asks = history[f'{prefix}_asks']
    
    init_idx = max(0, len(asks) - scope)
    vals = np.array(asks[init_idx : len(asks)])
    
    if scope > len(vals):
        padding = np.empty((scope - len(vals), 5))
        padding[:] = np.nan
        
        vals = np.concatenate((padding, vals))
        
    return vals

def sample_bids_volumes(
        history: Dict,
        scope: IntScope, 
        is_futures: Boolean
    ):
    prefix = 'etf' if not is_futures else 'hedge'
    
    bids = history[f'{prefix}_bids_volumes']
    
    init_idx = max(0, len(bids) - scope)
    vals = np.array(bids[init_idx : len(bids)])
    
    if scope > len(vals):
        padding = np.empty((scope - len(vals), 5))
        padding[:] = np.nan
        
        vals = np.concatenate((padding, vals))
        
    return vals

def sample_asks_volumes(
        history: Dict,
        scope: IntScope, 
        is_futures: Boolean
    ):
    prefix = 'etf' if not is_futures else 'hedge'
    
    asks = history[f'{prefix}_asks_volumes']
    
    init_idx = max(0, len(asks) - scope)
    vals = np.array(asks[init_idx : len(asks)])
    
    if scope > len(vals):
        padding = np.empty((scope - len(vals), 5))
        padding[:] = np.nan
        
        vals = np.concatenate((padding, vals))
        
    return vals

def sample_mid_prices(
        history: Dict,
        scope: IntScope, 
        is_futures: Boolean
    ):
    prefix = 'etf' if not is_futures else 'hedge'
    
    mids = history[f'{prefix}_mid_prices']
    
    init_idx = max(0, len(mids) - scope)
    vals = np.array(mids[init_idx : len(mids)])
    
    if scope > len(vals):
        padding = np.empty((scope - len(vals),))
        padding[:] = np.nan
        
        vals = np.concatenate((padding, vals))
        
    return vals
    