# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:22:40 2017

@author: beats
"""

from time import time

import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning


def train_dictionary(imgs, n_components=100, train_percentage = 0.8): 
    '''
    learns dictionary consisting of n_components atoms of size patch_size of input images
    
    Parameters
    ----------
    images: Input Images to learn the Dictionary.
    
    n_components: Number of atoms of the dictionary
    
    patch_size: Size of atoms
    
    train_percentage: Percentage of data on which the dictionary is trained from first to last element.
    
    Return
    ------
    Returns the Dictionary V.
    
    '''
# Extract Data from the images     
    rows, cols, timesteps, persons = imgs.shape    
   
    data = np.transpose(imgs, (2,0,1,3))
    print("data", data.shape)
    training_data = data[:,:,:,:int(np.floor(persons*train_percentage))]
    training_data = np.reshape(training_data, (timesteps, -1))
    training_data = training_data.T
 
    # Learn the Dictionary on the extracted patches
    print('Learning the Dictionary...')
    t0 = time()
    b_sz = 10
    dico = MiniBatchDictionaryLearning(alpha=0.5, n_iter=500, batch_size=b_sz, n_components = n_components, verbose=2)
    print("Training Data shape: ", training_data.shape)
 
    print("Tr:", training_data.shape)
 
    V = dico.fit(training_data).components_
    
    return dico, V

def test_dictionary(imgs, dico, V):
    '''
    Parameters
    ----------
    imgs: input images, should be some kind of undersampled
    V: Dictionary learned from train_dictionary
    
    '''
    rows, cols, timesteps, persons = imgs.shape
    
    testing_data = np.transpose(np.abs(imgs[:,:,:,0]), (2, 0, 1))
    testing_data = np.reshape(testing_data, (timesteps, -1))
    testing_data = testing_data.T
    code = dico.transform(testing_data)
    rec = np.dot(code, V)
    rec = np.reshape(rec.T, [timesteps, rows, cols])
    rec = np.transpose(rec, (1, 2, 0))
    img = rec
    return img

