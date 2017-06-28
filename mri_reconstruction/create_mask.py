# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:10:49 2017

@author: sbeat
"""

import numpy as np
import scipy as sp

def create_mask(undersampling, shape, method='uniform', sigma=30):
    '''
    Parameters:
    -----------
    undersampling: for now only used in the uniform undersampling
    shape: shape of image to be masked
    mask_type: 'uniform', 'gaussian' or 'poisson'
    '''
    # uniform distributed
    mu = shape[0]/2
    
    if method == 'uniform':

                       
    # Gaussian Mask
    if method == 'gaussian':
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

def create_poisson_disc_masks(a,b,c, *shape):
    