# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:39:26 2017

@author: beats@student.ethz.ch

Versions:
--------
Anaconda Python 2.7
Scikit-learn 0.19dev0

"""

import numpy as np
import scipy as sp
import random
import time

from sklearn.decomposition import MiniBatchDictionaryLearning

import poisson_disc_light
import enhanced_grid

import matplotlib.pyplot as plt

def normalize(imgs):
    print"normalizing..."
    imgs = imgs / 255
    #imgs_spat_median = np.median(imgs, axis=(0,1), keepdims=True)
    imgs_spat_std = np.std(imgs, axis=(0,1), keepdims=True)
    imgs_spat_std[imgs_spat_std < 1e-5] = 1e-5
    imgs_tmp_median = np.median(imgs, axis = 2, keepdims=True)
    
    #imgs -= imgs_spat_median
    imgs /= imgs_spat_std
    imgs -= imgs_tmp_median
    
    imgs[np.isnan(imgs)]=0
    
    return imgs
    
def fft_transform(imgs):
    print"transform to k-space..."
    k_imgs = np.fft.fft2(imgs, axes=(0,1))
    k_imgs = np.fft.fftshift(k_imgs)
    return k_imgs

def inverse_fft_transform(k_imgs):
    print"transforming back from k-space..."
    imgs = np.real(np.fft.ifft2(np.fft.ifftshift(k_imgs), axes=(0,1)))
    return imgs

def mask_imgs(imgs, fname, method='uniform', p_zeros=0.5):
    print"masking imgs..."
    
    if method == 'uniform':
        masks = np.zeros((imgs.shape))
        for t in range(imgs.shape[2]):
            for p in range(imgs.shape[3]):
                masks[:,:,t,p] = np.random.choice((0,1), size=(imgs.shape[0],imgs.shape[1]), p=(1-p_zeros, p_zeros))
    
    
    t0 = time.time()
    for i in range(imgs.shape[3]):
        for j in range(imgs.shape[2]):
            imgs[:,:,j,i] *= masks[:,:,j,i]
    print"done in %.1fs." %(time.time()-t0)
    return imgs

def imgs_to_data(imgs, n_components):
    print"converting imgs to data..."
    data = np.transpose(imgs, (2,0,1,3))
    data = np.reshape(data, (n_components, -1), order='F')
    data = data.T
    return data

def data_to_imgs(data, shape):
    print"converting data to imgs..."
    data = data.T
    imgs = np.reshape(data, (shape[2], shape[0], shape[1], -1), order='F')
    imgs = np.transpose(imgs,(1,2,0,3))
    return imgs

def imgs_error(img1, img2):
    err = np.sqrt(np.sum((img1 - img2) ** 2))
    err /= (img1.shape[0] * img1.shape[1])
    return err

def total_error(imgs_ref, imgs_rec):
    err_tot = 0
    for i in range(imgs_ref.shape[3]):
        for j in range(imgs_ref.shape[2]):
            err_tot += imgs_error(imgs_ref[:,:,j,i], imgs_rec[:,:,j,i])
    return err_tot

def sparsity(A):
    A = np.ravel(A)
    return 1-float(np.count_nonzero(A))/len(A)

def get_data(imgs, train_amount=0.8, test_amount=0.2, undersampling=0.5, normalize=True):
    #get random 80% as training images
    np.random.choice()
    #get the rest as testing images and process them
    
    k_test = fft_transform(imgs_test)
    k_mskd = msks*k_test
    imgs_test = inverse_fft_transform(k_mskd)
    #get imgs_test_ref
    
    #normalize all chunks
    if normalize:
        imgs_train = normalize(imgs_train)
        imgs_test = normalize(imgs_test)
        imgs_test_ref = normalize(imgs_test_ref)
    
    
    return imgs_train, imgs_test, imgs_test_ref

def initialize_dictionary(n_components, data_train):
    print"initializing dictionary..."
    init = np.zeros((n_components, data_train.shape[1]))
    
    for k in range(int(n_components/2)):
        for n in range(data_train.shape[1]):
            init[k,n] = np.cos((np.pi/data_train.shape[1])*n*(k+0.5))
            
    for i in range(int(n_components/2),n_components):
        init[i,:] = data_train[int(random.uniform(0,data_train.shape[0])),:]
    
    return init

#==============================================================================
# Main:
#==============================================================================
# Variables
n_components = 40
undersampling = np.linspace(0,1, num=10)
err_diff = np.zeros(undersampling.shape)

# Read
imgs = sp.io.loadmat('ismrm_ssa_imgs.mat', 'imgs')
imgs = np.float64(imgs)

# Preprocess
for i in range(len(undersampling)):
    imgs_train, imgs_test, imgs_test_ref = get_data(imgs, train_amount=0.8, test_amount=0.2, 
                                                    undersampling=undersampling[i], normalize=True)
    data_train = imgs_to_data(imgs_train)
    data_test = imgs_to_data(imgs_test)
    
    init = initialize_dictionary(n_components, data_train)
    
    # learn
    dico = MiniBatchDictionaryLearning(n_components=n_components, alpha=0.2, alpha_transform=0.5,
                                       n_iter=250, batch_size=1000, dict_init=None, verbose=0,
                                       fit_algorithm='lars', transform_algorithm='lasso_lars')
    dico.fit(data_train)
    V = dico.components_
    
    # Test
    code = dico.transform(data_test)
    recs = np.dot(code, V)
    recs = data_to_imgs(recs)
    
    err_diff[i] = total_error(imgs_test_ref, recs) - total_error(imgs_test_ref, imgs_test)
    code_sparsity = sparsity(code)



