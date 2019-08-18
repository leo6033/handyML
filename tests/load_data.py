#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:44:03 2019

@author: soushigou
"""

import pandas as pd

def load_data(str):
    try:
        data=pd.read_csv("../dataset/"+str)
    except:
        traceback.print_exc()
        
    return data
