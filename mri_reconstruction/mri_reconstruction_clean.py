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

def normalize_imgs(imgs):
    print"normalizing..."
    imgs = imgs / 255
#   imgs_spat_median = np.median(imgs, axis=(0,1), keepdims=True)
    imgs_spat_std = np.std(imgs, axis=(0,1), keepdims=True)
    imgs_spat_std[imgs_spat_std < 1e-5] = 1e-5
    imgs_tmp_median = np.median(imgs, axis = 2, keepdims=True)
    
#   imgs -= imgs_spat_median
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

def mask_imgs(imgs, method='uniform', p_zeros=0.5):
    print"masking imgs..."
    
    if method == 'uniform':
        masks = np.zeros((imgs.shape))
        for t in range(imgs.shape[2]):
            for p in range(imgs.shape[3]):
                masks[:,:,t,p] = np.random.choice((0,1), size=(imgs.shape[0],imgs.shape[1]), p=(p_zeros, 1-p_zeros))
    
    
    t0 = time.time()
    for i in range(imgs.shape[3]):
        for j in range(imgs.shape[2]):
            imgs[:,:,j,i] *= masks[:,:,j,i]
    print"done in %.1fs." %(time.time()-t0)
    return imgs

def imgs_to_data(imgs, n_components):
#   expects a 4 dimensional imgs parameter, make it more generig so a 2D img can be passed
    print"converting imgs to data..."

    if len(imgs.shape) < 4:
        data = np.transpose(imgs, (2,0,1))
        data = np.reshape(data, (n_components, -1), order='F')
        data = data.T
        return data
    
    data = np.transpose(imgs, (2,0,1,3))
    data = np.reshape(data, (n_components, -1), order='F')
    data = data.T
    return data

def data_to_imgs(data, shape):
#   expects a 4 dimensional imgs parameter, make it more generig so a 2D img can be passed
    print"converting data to imgs..."
    
    data = data.T
    
    if data.shape[0] == shape[2] * shape[0]**2:
        imgs = np.reshape(data, (shape[2], shape[0], shape[1]), order='F')
        return imgs
        
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

def get_data(imgs, undersampling=0, train_amount=0.8, normalize=True):
    
#   Initialization
    imgs_train = np.empty((imgs.shape[0], imgs.shape[1], imgs.shape[2], int(imgs.shape[3]*train_amount)))
    imgs_test = np.empty((imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3]-int(imgs.shape[3]*train_amount)))
    imgs_test_ref = imgs_test.copy()
    
#   Get random 80% of the images as training images
    ind_train = np.random.choice(imgs.shape[3], int(imgs.shape[3]*train_amount), replace=False)
    for i in range(len(ind_train)):
        imgs_train[:,:,:,i] = imgs[:,:,:,ind_train[i]]
        
#   Get the remaining images as testing images and process them
    ind_test = range(imgs.shape[3])
    for i in range(len(ind_train)):
        ind_test.remove(ind_train[i])
    
    for i in range(len(ind_test)):
        imgs_test_ref[:,:,:,i] = imgs[:,:,:,ind_test[i]]
    
    k_test = fft_transform(imgs_test_ref)
    k_mskd = mask_imgs(k_test, method='uniform', p_zeros=undersampling)
    imgs_test = inverse_fft_transform(k_mskd)
    
#    normalize all sets
    if normalize:
        imgs_train = normalize_imgs(imgs_train)
        imgs_test = normalize_imgs(imgs_test)
        imgs_test_ref = normalize_imgs(imgs_test_ref)
    
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
undersampling = 0.5
train_amount = 0.8

# Training Parameters
batch_size = 1000
n_iter = 250

# Algorithms
fit_algorithm='lars'
transform_algorithm='lasso_lars'

# Sparsity Parameter
alpha_train = 0.2
alpha_test = 0.5
# Output amount
verbose = 0


# Read
imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
imgs = np.float64(imgs['imgs'])


# Preprocess

imgs_train, imgs_test, imgs_test_ref = get_data(imgs, train_amount=train_amount, undersampling=undersampling)

imgs_test = imgs_test[:,:,:,:4]
imgs_train = imgs_train[:,:,:,:2]

data_train = imgs_to_data(imgs_train, n_components)
data_test = imgs_to_data(imgs_test, n_components)

init = initialize_dictionary(n_components, data_train)


# Train
dico = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha_train,
                                   n_iter=n_iter, batch_size=batch_size, dict_init=init, verbose=verbose,
                                   fit_algorithm=fit_algorithm, transform_algorithm=transform_algorithm)
dico.fit(data_train)
V = dico.components_


# Test
dico.alpha_transform = alpha_test
code = dico.transform(data_test)
recs = np.dot(code, V)
recs = data_to_imgs(recs, imgs_test.shape)


#code_sparsity = sparsity(code)


# Plot
plt.plot()
plt.title("")
plt.xlabel("")
plt.ylabel("PSNR")

plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(imgs_test_ref[:,int(imgs_test_ref.shape[1]/2),:,0])
plt.title("Ground")
plt.xlabel('time')
plt.ylabel('y')
plt.subplot(1,3,2)
plt.imshow(imgs_test[:,int(imgs_test.shape[1]/2),:,0])
plt.title("Aliased")
plt.xlabel('time')
plt.ylabel('y')
plt.subplot(1,3,3)
plt.imshow(recs[:,int(recs.shape[1]/2),:,0])
plt.title("Reconstruction")
plt.xlabel('time')
plt.ylabel('y')