# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:22:40 2017

@author: beats
"""

import numpy as np

from sklearn.decomposition import MiniBatchDictionaryLearning

def train_dictionary(imgs, n_components): 
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
    #print("data", data.shape)
    data_train = data[:,:,:,:int(np.floor(persons))]
    data_train = np.reshape(data_train, (timesteps, -1))
    data_train = data_train.T
    #print("Training Data shape: ", data_train.shape)
    
    # Initialize dictionary as DCT
    #init = np.zeros([timesteps,n_components])
    #for i in range(n_components):
    #    for j in range(timesteps):
    #        init[j,i] = np.cos(np.pi/timesteps*(j+0.5)*i)

    dico = MiniBatchDictionaryLearning(n_components, alpha=0.5, n_iter=500, batch_size=1000, verbose=2) # 2Do: train function more generic
    V = dico.fit(data_train).components_
    
    return dico, V

def test_dictionary(imgs_under, dico, V):
    '''
    Parameters
    ----------
    imgs_under: undersampled reconstructed images
    imgs_ref: reference images
    dico: dico class
    V: Dictionary learned from train_dictionary
    test_percentage: last percent to test the dictionary V
    
    '''
    rows, cols, timesteps, persons = imgs_under.shape

    data_test = np.transpose(imgs_under, (2,0,1,3))
    #data_test = np.transpose(np.abs(imgs_under[:,:,:,10], (2, 0, 1))
    data_test = np.reshape(data_test, (timesteps, -1))
    data_test = data_test.T
    code = dico.transform(data_test)
    rec = np.dot(code, V)
    rec = np.reshape(rec.T, [timesteps, rows, cols, -1])
    rec = np.transpose(rec, (1,2,0,3))
    
    return rec



