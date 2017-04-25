# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:55:50 2017

@author: beats
"""

import numpy as np
import scipy as sp
import random
import time
import os.path as path

from sklearn.decomposition import MiniBatchDictionaryLearning

import poisson_disc_light
import enhanced_grid

import matplotlib.pyplot as plt

class mri_reconstruction:
    '''
    Class to reconstruct MRI Images with dictionary learning on large data sets of MRI images (128,128,25,375)
    '''
    
    def __init__(self):
        self.imgs = self.import_imgs()
        self.imgs_shape = self.imgs.shape
        self.rows = self.imgs.shape[0]
        self.cols = self.imgs.shape[1]
        self.timesteps = self.imgs.shape[2]
        self.persons = self.imgs.shape[3]
        
    def import_imgs(self):
        print('importing images...')
        imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
        imgs = np.float64(imgs['imgs'])
        return imgs
        
    def normalize(self, imgs):
        print('normalizing...')
        imgs_spat_median = np.median(imgs, axis=(0,1), keepdims=True)
        imgs_spat_std = np.std(imgs, axis=(0,1), keepdims=True)
        imgs_spat_std[imgs_spat_std < 1e-5] = 1e-5
        imgs_tmp_mean = np.mean(imgs, axis = 2, keepdims=True)
        
        imgs -= imgs_spat_median
        imgs /= imgs_spat_std
        imgs -= imgs_tmp_mean
        return imgs
    
    def transform(self, imgs):
        print('transform to k-space...')
        k_imgs = np.fft.fft2(imgs, axes=(0,1))
        k_imgs = np.fft.fftshift(k_imgs)
        return k_imgs
    
    def inverse_transform(self,k_imgs):
        print('transforming back from k-space...')
        imgs = np.real(np.fft.ifft2(np.fft.ifftshift(k_imgs), axes=(0,1)))
        return imgs
    
    def create_r_grid(self, a, b, c):
        w = self.cols
        h = self.rows
        
        r_grid = enhanced_grid.Grid2D((w, h))
        center = (w/2, h/2)

        for index in r_grid.index_iter():
            r_grid[index] = a-(a*np.e**(-(poisson_disc_light.dist(index, center)-b)**2/(2*c**2))) + 0.1 # avoid 0 radius!
        return r_grid
    
    def create_mask(self, a, b, c, *shape):
        
        r_grid = self.create_r_grid(a,b,c)
        
        p = poisson_disc_light.sample_poisson(self.cols, self.rows, r_grid, 30)
    
        mask = np.zeros((self.cols,self.rows))
        for item in p:
            mask[item]=1
        sp.io.savemat('masks.mat', {'masks':mask})
        
        pass
    
    def mask_imgs(self, a, b, c, imgs, masks):
        # only mask the testing data since masking all images would take long
        print('masking imgs...')
        try:
            print('importing masks...')
            imgs = sp.io.loadmat('masks.mat')
            imgs = np.float64(imgs['masks'])
            
        except:
            print('Error while importing masks')
            return False
        
        t0 = time.time()
        for i in range(imgs.shape[3]):
            for j in range(imgs.shape[2]):
                imgs[:,:,j,i] *= masks[:,:,j,i]
        print("done in %.1fs." %(time.time()-t0))
        return imgs
    
    def imgs_to_data(self, imgs):
        print('converting imgs to data...')
        data = np.transpose(imgs, (2,0,1,3))
        data = np.reshape(data, (self.timesteps, -1))
        data = data.T
        return data
    
    def data_to_imgs(self, data):
        print('converting data to imgs...')
        data = data.T
        imgs = np.reshape(data, (self.timesteps, self.cols, self.rows, -1))
        imgs = np.transpose(imgs,(1,2,0,3))
        return imgs
    
    def initialize_dictionary(self, n_components, data_train):
        '''
        2Do
        ----
        initialize half with DCT, half with random training vectors
        '''
        print('initializing dictionary...')
        init = np.ndarray((n_components, self.timesteps))
        
        for j in range(int(n_components/2)):
            init[j,:] = np.cos(np.pi/init.shape[1]*(j+0.5))
        for i in range(int(n_components/2), n_components):
            init[i,:] = data_train[int(random.uniform(0,data_train.shape[0])),:]
        return init
    
    def get_training_data(self):
        return
    
    def get_testing_data(self):
        return
    
    def train_dictionary(self, imgs_train, n_components, alpha, n_iter, batch_size, verbose):
        # normalize images
        imgs_norm = self.normalize(imgs_train)
        # transform to (samples, features) shape
        data_train = self.imgs_to_data(imgs_norm)
        print("training dictionary...")
        t0 = time.time()
        self.dico = MiniBatchDictionaryLearning(n_components, alpha, n_iter,
                                                fit_algorithm='lars', n_jobs=1,
                                                batch_size=1000, verbose=2)
        self.V = self.dico.fit(data_train).components_
        print("done in %.1fs." %(time.time()-t0))
        return
    
    def test_dictionary(self, imgs_test):
        imgs_norm = self.normalize(imgs_test)
        data_test = self.imgs_to_data(imgs_norm)
        print('testing dictionary...')
        t0 = time.time()
        self.code = self.dico.transform(data_test)
        recs = np.dot(self.code,self.V)
        recs = self.data_to_imgs(recs)
        print("done in %.1fs." %(time.time()-t0))
        return recs
    
    def imgs_error(self, img1, img2):
        err = np.sqrt(np.sum((img1 - img2) ** 2))
        err /= (img1.shape[0] * img1.shape[1])
        return err
    
    def total_error(self, imgs_ref, imgs_rec):
        err_tot = 0
        for i in range(imgs_ref.shape[3]):
            for j in range(self.timesteps):
                err_tot += self.imgs_error(imgs_ref[:,:,j,i], imgs_rec[:,:,j,i])
        return err_tot
    
    def error_difference(self, imgs, imgs_rec, imgs_rec_dic):
        return self.total_error(imgs,imgs_rec) - self.total_error(imgs,imgs_rec_dic)
    
    def sparsity(self, A):
        A = np.ravel(A)
        return round(1-float(np.count_nonzero(A))/len(A), 2)
    
    def minimize_alpha(self):
        return
    
    def create_plots(self):
        plt.figure(figsize=(20,20))
        plt.subplot(1,4,1)
        plt.imshow(ref_imgs_test[:,:,0,0], cmap='gray')
        plt.subplot(1,4,2)
        plt.imshow(np.log(k_mskd[:,:,0,0]), cmap = 'gray')
        plt.subplot(1,4,3)
        plt.imshow(imgs_test[:,:,0,0], cmap= 'gray')
        plt.subplot(1,4,4)
        plt.imshow(recs[:,:,0,0], cmap = 'gray')
        pass
      
obj = mri_reconstruction()
    
# Training amount
percentage = 0.8

# Gauss curve parameters for masking: a = radius, b = offset, c = std deviation
a = 100
b = 0
c = 300

# Training
imgs_train = obj.imgs[:,:,:,:int(percentage*obj.persons)]
obj.train_dictionary(imgs_train, n_components=50, alpha=0.005, n_iter=250, batch_size=1000, verbose=2)

# Testing
ref_imgs_test = obj.imgs[:,:,:,int(percentage*obj.persons):obj.persons]
k_test = obj.transform(ref_imgs_test)
k_mskd = obj.mask_imgs(a, b, c, k_test, masks, verbose=True)
imgs_test = obj.inverse_transform(k_mskd)
recs = obj.test_dictionary(imgs_test)

# calculate error
err_diff = obj.error_difference(ref_imgs_test, imgs_test, recs)
print(err_diff)
#printing
plt.figure(figsize=(20,20))
plt.subplot(1,4,1)
plt.imshow(ref_imgs_test[:,:,0,0])
plt.subplot(1,4,2)
plt.imshow(np.log(k_mskd[:,:,0,0]))
plt.subplot(1,4,3)
plt.imshow(imgs_test[:,:,0,0])
plt.subplot(1,4,4)
plt.imshow(recs[:,:,0,0])
