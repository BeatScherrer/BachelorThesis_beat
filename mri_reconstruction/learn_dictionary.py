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
    # import matlab data (johannes')
    try:
        imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
        imgs = imgs['imgs']
        imgs = np.float64(imgs)
    except:
        print('Error while loading images!')

    rows, cols, timesteps, persons = imgs.shape
    
    train_percentage = 0.1

    # Extract Data from the images        
    data = np.transpose(imgs, (2,0,1,3))
    training_data = data[:,:,:,:int(np.floor(persons*train_percentage))]
    training_data = np.reshape(training_data, (timesteps, -1))
    training_data = training_data.T
     
    # Learn the Dictionary on the extracted patches
    print('Learning the Dictionary...')
    t0 = time()
    b_sz = 10
    dico = MiniBatchDictionaryLearning(40, alpha=1, n_iter=10, batch_size=b_sz, verbose=1)
    print("Training Data shape: " + str(training_data.shape[0]) + " " + str(training_data.shape[1]))
    for i in range(int(training_data.shape[0]/b_sz)):
        print("batch %d of %d"%(i, int(training_data.shape[0]/b_sz)))
        batch = training_data[i*b_sz : (i+1)*b_sz, :]
        V = dico.partial_fit(batch).components_
        if i > 200:
            break
    dt = time() - t0
    print('Dictionary learned in %.1fs' %dt)
    return dico,V

def test_dictionary(imgs, dico, V):
    '''
    Parameters
    ----------
    imgs: input images, should be some kind of undersampled
    V: Dictionary learned from train_dictionary
    
    '''
    rows, cols, timesteps, persons = imgs.shape
    
    testing_data = np.abs(imgs[:,:,:,300])
    testing_data = np.reshape(testing_data, (timesteps, -1))
    testing_data = testing_data.T
    code = dico.transform(testing_data)
    rec = np.dot(code, V)
    img = np.reshape(rec, [rows, cols, timesteps])
    return img


