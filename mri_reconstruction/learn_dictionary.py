# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:22:40 2017

@author: beats
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version

def train_dictionary(images, n_components=100, patch_size=(5,5), train_percentage = 0.8): 
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
    return V

def test_dictionary(images, n_components, transform_algorithms, patch_size=(5,5), test_percentage=0.2):
    '''
    '''
    return


############################# Testing ##########################################

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
training_data = data[:,:,:,:np.floor(persons*train_percentage)]
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


testing_data = np.abs(reconstructions_undersampled[:,:,:, 300])
testing_data = np.reshape(testing_data, (timesteps, -1))
testing_data = testing_data.T
code = dico.transform(testing_data)
rec = np.dot(code, V)
img = np.reshape(rec, [rows, cols, timesteps])
print('Dictionary trained on %d vectors in %.2fs.' % (len(training_data), dt))
