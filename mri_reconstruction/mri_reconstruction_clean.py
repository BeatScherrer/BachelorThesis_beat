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

from sklearn.decomposition import MiniBatchDictionaryLearning

#import poisson_disc_light
#import enhanced_grid

import matplotlib.pyplot as plt

def normalize(imgs):
    print"normalizing..."
    imgs = imgs / 2000
    imgs_spat_median = np.median(imgs, axis=(0,1), keepdims=True)
    imgs_spat_std = np.std(imgs, axis=(0,1), keepdims=True)
    imgs_spat_std[imgs_spat_std < 1e-5] = 1e-5
    imgs_tmp_median = np.median(imgs, axis = 2, keepdims=True)
    
    imgs -= imgs_spat_median
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

def create_masks(shape, undersampling, method, full_center=False, k=8, export_masks=False):
    
    k = int(k/2)
    masks = np.zeros((shape))
    mask = np.zeros(shape[:2])
    
    if method == 'uniform':
        for t in range(shape[2]):
            for p in range(shape[3]):
                masks[:,:,t,p] = np.random.choice((0,1), size=(shape[:2]), p=(undersampling, 1-undersampling))
        
    if method == 'uniform_lines':
        for t in range(shape[2]):
            for p in range(shape[3]):
                mask = np.random.choice((0,1), size=(shape[0]), p = (undersampling, 1-undersampling))
                for i in range(shape[1]):
                    masks[:,i,t,p] = mask
    
    if method == 'gaussian_lines':
        threshhold = sp.stats.norm.ppf(undersampling)
        for t in range(shape[2]):
            for p in range(imgs.shape[3]):
                mask = np.random.randn(shape[0])
                for i in range(len(mask)):
                    if mask[i] < threshhold:
                        mask[i] = 0
                    else:
                        mask[i] = 1  
                for j in range(imgs.shape[1]):
                    masks[:,j,t,p] = mask
    
    if full_center:
        for t in range(shape[2]):
            for p in range(shape[3]):
                masks[int(shape[0]/2)-k:int(shape[0]/2)+k,:,t,p] = 1
        
    if export_masks:
        sp.io.savemat('masks' + '_' + str(method) + '_' + str(undersampling) + '.mat', {'masks':masks})
        
    return masks

def mask_imgs(imgs, method='uniform', full_center=False, k=8, undersampling=0.5, n_gauss=100, variance=20, return_masks=False, export=False):
    '''
    parameters
    ----------
    imgs: input images
    method: defines subsampling method
    full_center: bool if the center of the mask should be fully sampled or not
    k: amount of  frequencies to fully sample from center
    '''
    print"masking imgs..."
    
    masks = create_masks(imgs.shape, undersampling, method, full_center=full_center, k=k)
    
    for i in range(imgs.shape[3]):
        for j in range(imgs.shape[2]):
            imgs[:,:,j,i] *= masks[:,:,j,i]
    
    if return_masks:
        return imgs, masks
    
    return imgs

def imgs_to_data(imgs, n_features):

    if len(imgs.shape) < 4:
        data = np.transpose(imgs, (2,0,1))
        data = np.reshape(data, (n_features, -1), order='F')
        data = data.T
        return data
    
    data = np.transpose(imgs, (2,0,1,3))
    data = np.reshape(data, (n_features, -1), order='F')
    data = data.T
    return data

def data_to_imgs(data, shape):
#   expects a 4 dimensional imgs parameter, make it more generig so a 2D img can be passed

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

def peak_signal_to_noise_ratio(imgs_ref, recs):
    '''
    '''
    s = imgs_ref.shape
    P = s[0]*s[1]*s[2]
    
    err = 0
    err = np.linalg.norm(imgs_ref - recs)
    
    psnr = 10*np.log10(1/(err/P))
        
    return psnr

def sparsity(A):
    A = np.ravel(A)
    return 1-np.float64(np.count_nonzero(A))/len(A)

def get_imgs(imgs, undersampling=0, train_amount=0.8):
    
#   Initialization
    imgs_train = np.empty((imgs.shape[0], imgs.shape[1], imgs.shape[2], int(imgs.shape[3]*train_amount)))
    imgs_test_ref = np.empty((imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3]-int(imgs.shape[3]*train_amount)))
    
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
    
    return imgs_train, imgs_test_ref

def initialize_dictionary(n_components, data_train):
    if verbose:
        print"initializing dictionary..."
    init = np.zeros((n_components, data_train.shape[1]))
    
    for k in range(int(n_components/2)):
        for n in range(data_train.shape[1]):
            init[k,n] = np.cos((np.pi/data_train.shape[1])*n*(k+0.5))
            
    for i in range(int(n_components/2),n_components):
        init[i,:] = data_train[int(random.uniform(0,data_train.shape[0])),:]
    
    return init

def export_as_mat(a,b, undersamp, n_comp, b_s, n_iter, fit_alg, transf_alg, alpha_train, alpha_test, export_info=False):
    info = ('n_comp=' + str(n_comp) + ' batch_size=' + str(b_s) + 
            ' n_iter=' + str(n_iter) + ' fit_alg=' + str(fit_alg) + 
            ' transform_alg=' + str(transf_alg) + ' alpha_train=' + 
            str(alpha_train) + ' alpha_test=' + str(alpha_test))
    sp.io.savemat(str(a) + '_' + str(b) + '.mat', {str(a):a, str(b):b, 'info':info})
    return

def plot_mat(fname):
    return

#==============================================================================
# Variables
n_components = 40
undersampling = 0.5
train_amount = 0.8

# Training Parameters
batch_size = 1000
n_iter = 250

n_jobs = 1

# Algorithms
fit_algorithm='lars'
transform_algorithm='lasso_lars'

# Sparsity Parameter
alpha_train = 0.2
alpha_test = 0.5

# Output amount
verbose = 2

# Plot Variables:
resolution=10

alpha_test_vec = np.linspace(0, 1, num=resolution)
#alpha_test = np.linspace(0,1,num=resolution)
#undersampling = np.linspace(0,1,num=resolution)
#n_gauss_plot = np.linspace(0,100, num=resolution)
spars=np.zeros(resolution)
psnr = np.zeros(resolution)

#==============================================================================
# main  algorithm
#==============================================================================

# Read
imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
imgs = np.float64(imgs['imgs'])

# Preprocess
imgs_train, imgs_test_ref = get_imgs(imgs, train_amount=train_amount)

# Select Testing and training set size    
imgs_test_ref = imgs_test_ref[:,:,:,:]
imgs_train = imgs_train[:,:,:,:]

imgs_train = normalize(imgs_train)
imgs_test_ref = normalize(imgs_test_ref)

data_train = imgs_to_data(imgs_train, imgs.shape[2])

init = initialize_dictionary(n_components, data_train)

for i in range(resolution):
    alpha_test = alpha_test_vec
    # Train
    dico = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha_train,
                                       n_iter=n_iter, n_jobs=n_jobs, batch_size=batch_size, dict_init=init.copy(), verbose=verbose,
                                       fit_algorithm=fit_algorithm, transform_algorithm=transform_algorithm)
    print"fitting data..."
    #for j in range(int(data_train.shape[0]/dico.batch_size)):
    #    data_batch = data_train[dico.batch_size*j:dico.batch_size*(j+1)]
    #    dico.partial_fit(data_batch)
    dico.fit(data_train)
    
    V = dico.components_
    
    # Test
    k_test = fft_transform(imgs_test_ref)
    k_mskd, masks = mask_imgs(k_test, method='uniform', full_center=True, k=8, 
                              undersampling=undersampling, n_gauss=100, variance=30, 
                              return_masks=True)
    imgs_test = inverse_fft_transform(k_mskd)
    imgs_test = normalize(imgs_test)
    data_test = imgs_to_data(imgs_test, imgs.shape[2])
    
    dico.transform_alpha = alpha_test
    print"encoding data..."
    code = dico.transform(data_test)
    recs = np.dot(code, V)
    recs = data_to_imgs(recs, imgs_test.shape)
    
    psnr[i] = peak_signal_to_noise_ratio(imgs_test_ref, recs)

#code_sparsity = sparsity(code)
#export_as_mat(alpha_train_vec, psnr, undersampling, n_components, batch_size, 
#              n_iter, fit_algorithm, transform_algorithm, alpha_train, alpha_test, 
#              export_info=True)


#==============================================================================
# Plots
#==============================================================================
#psnr
plt.plot(alpha_test_vec, psnr, marker='x')
plt.title("")
plt.xlabel("undersampling factor")
plt.ylabel("PSNR")

#chain
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(imgs_test_ref[:,:,0,0])
plt.title('ground')
plt.subplot(1,3,2)
plt.imshow(imgs_test[:,:,0,0])
plt.title('test image')
plt.subplot(1,3,3)
plt.imshow(recs[:,:,0,0])
plt.title('reconstruction')

#slice through time
plt.figure(figsize=(10,10))
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