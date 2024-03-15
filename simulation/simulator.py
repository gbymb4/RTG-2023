# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:26:27 2023

@author: Gavin
"""

from typing import Callable, Union, Tuple, Dict

from ._simio import load_history
from ._config import POSITION_LIMIT, MAX_VOLUME, FEE
from ._util import (
    next_instrument, opposite_side,
    sample_history, etf_to_future_side,
    min_price, max_price,
    protected_volume, compute_holdings_value
)


from ready_trader_go import Instrument, Side

class Simulator:
    
    def __init__(self, signal_strategy: Callable, signal_threshold: float=10) -> None:
        self.history = load_history()
        
        self.signal_strategy = signal_strategy   
        self.signal_threshold = signal_threshold
        
        self.tick = self.etf_lots = self.hedge_lots = 0
        self.instrument = Instrument.ETF
        
        self.bought = self.sold = 0
        
    
    
    def run(self, _ticks: Union[int, None]=None) -> Dict:
        max_tick = len(next(iter(self.history.values()))) if _ticks is None else _ticks
        
        for _ in range(max_tick):
            self.update()
        
        return {}
    
        
        
    def update(self) -> None:
        if self.instrument == Instrument.FUTURE:
            history = sample_history(self.history, self.tick)
            
            signal = self.signal_strategy(history)
            
            etf_side = self.decipher_signal(signal)
            
            if etf_side is not None:
                hedge_side = opposite_side(etf_to_future_side(etf_side))
                
                etf_price, etf_volume = self.price_volume_strategy(
                    history, 
                    etf_side, 
                    Instrument.ETF
                )
                
                hedge_price, _ = self.price_volume_strategy(
                    history, 
                    opposite_side(etf_side), 
                    Instrument.FUTURE
                )
                
                etf_volume = protected_volume(
                    etf_volume, 
                    etf_side, 
                    self.etf_lots, 
                    POSITION_LIMIT
                )
                
                self.place_order(etf_side, etf_price, etf_volume)
                self.place_hedge(hedge_side, hedge_price, etf_volume)
        
            self.tick += 1 
            
        self.instrument = next_instrument(self.instrument)
        
        
        
    def decipher_signal(self, signal: float) -> Union[Side, None]:
        if signal > self.signal_threshold:
            return Side.BUY
        if signal < -self.signal_threshold:
            return Side.SELL
        
        return None
    
    
    
    def price_volume_strategy(
            self, 
            history: Dict,
            side: Side,
            instrument: Instrument
        ) -> Union[Tuple[int, int], None]:
        
        prefix = 'etf' if instrument == Instrument.ETF else 'hedge'
        
        data = {key: value[-1] for key, value in history.items()}
        
        if side == Side.BUY:
            prices, volumes = data[f'{prefix}_asks'], data[f'{prefix}_asks_volumes']
            
            price, vol_available = max_price(prices, volumes)
            volume = min(MAX_VOLUME, vol_available)
        
            return price, volume
        
        if side == Side.SELL:
            prices, volumes = data[f'{prefix}_bids'], data[f'{prefix}_bids_volumes']
            
            price, vol_available = min_price(prices, volumes)
            volume = min(MAX_VOLUME, vol_available)
            
            return price, volume
        
        return None
    
    
        
    def place_order(self, side: Side, price: int, volume: int) -> None:
        if side == Side.BUY:
            self.etf_lots += volume
            self.bought += price * volume * (1 + FEE)
            
        if side == Side.SELL:
            self.etf_lots -= volume
            self.sold += price * volume * (1 - FEE)
            
            
    
    def place_hedge(self, side: Side, price: int, volume: int) -> None:
        if side == Side.BID:
            self.hedge_lots += volume
            self.bought += price * volume * (1 + FEE)
            
        if side == Side.ASK:
            self.hedge_lots -= volume
            self.sold += price * volume * (1 - FEE)
            
            
            
    def instantaneous_profit(self):
        data = {key: value[self.tick] for key, value in self.history.items()}
        
        traded_profit = self.sold - self.bought
        
        etf_holdings = compute_holdings_value(data, Instrument.ETF, self.etf_lots)
        hedge_holdings = compute_holdings_value(data, Instrument.FUTURE, self.hedge_lots)
        
        holdings_value = etf_holdings + hedge_holdings
        total_profit = traded_profit + holdings_value
        
        return total_profit
        