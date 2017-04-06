# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:22:40 2017

@author: beats
"""

from time import time

import numpy as np
import scipy as sp

from scipy.fftpack import dct
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
    print("Learning dictionary on first %.1f%% of data..." %(train_percentage*100))  
    t0 = time()
    rows, cols, timesteps, persons = imgs.shape    
   
    data = np.transpose(imgs, (2,0,1,3))
    #print("data", data.shape)
    data_train = data[:,:,:,:int(np.floor(persons*train_percentage))]
    data_train = np.reshape(data_train, (timesteps, -1))
    data_train = data_train.T
    #print("Training Data shape: ", data_train.shape)
    
    # Initialize dictionary as DCT
    init = dct(data_train)

    dico = MiniBatchDictionaryLearning(n_components, alpha=0.5, n_iter=500, dict_init = init, batch_size=10, verbose=0)
    V = dico.fit(data_train).components_
    
    dt = time() - t0
    print("Dictionary learned in %.1fs on %.1f samples." %(dt, data_train.shape[0]*train_percentage))
    return dico, V

def test_dictionary(imgs_under, imgs_ref, dico, V, test_percentage = 0.2):
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
    
    print("Testing the Dictionary on last %.1f%% of data..." %(test_percentage*100))
    data_test = np.transpose(np.abs(imgs_under[:,:,:,int(persons*(1-test_percentage)):persons]), (2,0,1,3))
    #data_test = np.transpose(np.abs(imgs_under[:,:,:,10], (2, 0, 1))
    data_test = np.reshape(data_test, (timesteps, -1))
    data_test = data_test.T
    code = dico.transform(data_test)
    rec = np.dot(code, V)
    rec = np.reshape(rec.T, [timesteps, rows, cols, -1])
    rec = np.transpose(rec, (1,2,0,3))
    
    # calculate RMSE
    error = mse(imgs_ref[:,:,:, int(persons*(1-test_percentage)):persons], rec)
    
    return rec, error

def mse(imageA, imageB):
    
   error = 0
   for i in range(imageA.shape[2]):
        for j in range(imageA.shape[3]):
            error += np.linalg.norm((imageA[:,:,i,j]-imageB[:,:,i,j]), 'fro')
   error /= (imageA.shape[2] * imageA.shape[3])
            
   return error

