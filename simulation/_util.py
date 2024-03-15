# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:32:34 2023

@author: Gavin
"""

from typing import Dict, Union, Tuple, List

from ready_trader_go import Instrument, Side

from ._config import FEE

def next_instrument(instrument: Instrument) -> Union[Instrument, None]:
    if instrument == Instrument.ETF:
        return Instrument.FUTURE
    if instrument == Instrument.FUTURE:
        return Instrument.ETF
    
    return None



def opposite_side(side: Side) -> Union[Side, None]:
    if side == Side.BID:
        return Side.ASK
    if side == Side.ASK:
        return Side.BID
    
    if side == Side.BUY:
        return Side.SELL
    if side == Side.SELL:
        return Side.BUY
    
    return None



def etf_to_future_side(side: Side) -> Union[Side, None]:
    if side == Side.BID:
        return Side.BUY
    if side == Side.ASK:
        return Side.SELL
    
    if side == Side.BUY:
        return Side.BID
    if side == Side.SELL:
        return Side.ASK
    
    return None



def sample_history(history: Dict, idx: int) -> Dict:
    sampled_hist = {key: value[:idx + 1] for key, value in history.items()}
    
    return sampled_hist



def min_price(prices: List[int], volumes: List[int]) -> Tuple[int, int]:
    return prices[0], volumes[0]



def max_price(prices: List[int], volumes: List[int]) -> Tuple[int, int]:
    return prices[0], volumes[0]



def protected_volume(volume: int, side: Side, lots: int, threshold: int) -> int:
    if side == Side.BUY or side == Side.BID:
        headroom = threshold - lots
        return min(headroom, volume)
    
    if side == Side.SELL or side == Side.ASK:
        headroom = threshold + lots
        return min(headroom, volume)
    
    
    
def compute_holdings_value(curr_data: Dict, instrument: Instrument, lots: int) -> float:
    prefix = 'etf' if instrument == Instrument.ETF else 'hedge'
    
    if lots < 0:
        price = curr_data[f'{prefix}_asks'][0]
        value = price * lots * (1 + FEE)
        
        return value
        
    if lots > 0:
        price = curr_data[f'{prefix}_bids'][0]
        value = price * lots * (1 - FEE)
        
        return value
        
    return 0