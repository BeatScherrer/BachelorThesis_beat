# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:10:49 2017

@author: sbeat
"""

import numpy as np

def create_mask(undersampling, shape, mask_type, sigma=30):
    '''
    Parameters:
    -----------
    undersampling: for now only used in the uniform undersampling
    shape: shape of image to be masked
    mask_type: 'uniform', 'gaussian' or 'poisson'
    '''
    # uniform distributed
    mu = shape[0]/2
    
    if mask_type == 'uniform':
        mask = np.random.choice([0, 1], size=shape[0], p=[undersampling, 1-undersampling])

    # Poisson Mask
    if mask_type == 'poisson':                     
        poisson = np.random.poisson(shape[0]/2,shape[0])
        mask = np.zeros(shape[0])
        for i in poisson:
            mask[int(i)] = 1
                       
    # Gaussian Mask
    if mask_type == 'gaussian':
        gaussian = sigma * np.random.randn(shape[0]) + mu
        mask = np.zeros(shape[0])
        for i in gaussian:
            if i > shape[0]-1:
                continue
            if i < 0:
                continue
            mask[int(i)]=1
                     
    mask = np.array([mask]*shape[1]).T
                         
    return mask