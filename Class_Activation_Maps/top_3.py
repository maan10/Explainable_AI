# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:12:32 2020

@author: Vanditha Rao

This script allows the user to obtain the top 3 predictions made by the model
for a given image

This script requires that tensorflow be installed within the python
environment you are running this script in.

Here, tensorflow version 2.2, and python version 3.7.7 is used. This file is
imported as a module 

"""

import tensorflow

def top_3(predicitons, k):
    
    prediciton_k_values, top_k_indices = tensorflow.nn.top_k(predicitons, k)
    prediciton_k_values, top_k_indices = prediciton_k_values.numpy(), top_k_indices.numpy()
    
    return top_k_indices, prediciton_k_values