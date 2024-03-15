# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:26:35 2023

@author: Gavin
"""

import json

def load_history():
    with open('out/history.json') as file:
        data = json.load(file)
        
    return data