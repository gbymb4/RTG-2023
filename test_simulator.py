# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:40:22 2023

@author: Gavin
"""

import random

from simulation import Simulator

def main():
    random.seed(0)
    
    def signal_strategy(_):
        return random.randint(-20, 20)
    
    sim = Simulator(signal_strategy)
    
    sim.run()
    
if __name__ == '__main__':
    main()