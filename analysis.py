# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:18:00 2023

@author: Gavin
"""

import json

import matplotlib.pyplot as plt

def main():
    with open('out/history.json', 'r') as file:
        data = json.load(file)
        
    data_points = (1, 100)
        
    xs = range(data_points[1] - data_points[0])
    
    plt.plot(xs, data['etf_mid_prices'][data_points[0] : data_points[1]], label='etf')
    plt.plot(xs, data['hedge_mid_prices'][data_points[0] : data_points[1]], label='hedge')
    
    print(data['etf_bids_volumes'])
    
    plt.legend()
    
    plt.show()
    
if __name__ == '__main__':
    main()