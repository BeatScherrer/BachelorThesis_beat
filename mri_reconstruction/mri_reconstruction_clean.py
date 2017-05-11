# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:55:50 2017

@author: beats

Version
------
Python 2.7
scikit-learn 0.19.dev0
"""
import numpy as np
import scipy as sp
import random
import time

from sklearn.decomposition import MiniBatchDictionaryLearning

import poisson_disc_light
import enhanced_grid

import matplotlib.pyplot as plt

class mri_reconstruction:
    '''
    Class to reconstruct MRI Images with dictionary learning on large data sets of MRI images (128,128,25,375)
    '''
    def __init__(self):
        self.imgs = self.import_imgs('ismrm_ssa_imgs.mat', 'imgs')
        self.imgs_shape = self.imgs.shape
        self.rows = self.imgs.shape[0]
        self.cols = self.imgs.shape[1]
        self.timesteps = self.imgs.shape[2]
        self.persons = self.imgs.shape[3]
        
    def import_imgs(self, fname, name):
        print"importing images..."
        imgs = sp.io.loadmat(fname)
        imgs = np.float64(imgs[name])
        return imgs
        
    def normalize(self, imgs):
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
    
    def transform(self, imgs):
        print"transform to k-space..."
        k_imgs = np.fft.fft2(imgs, axes=(0,1))
        k_imgs = np.fft.fftshift(k_imgs)
        return k_imgs
    
    def inverse_transform(self, k_imgs):
        print"transforming back from k-space..."
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
    
    def create_mask(self, a, b, c, shape):
        print"create_mask"
        r_grid = self.create_r_grid(a,b,c)
        
        p = poisson_disc_light.sample_poisson(shape[0], shape[1], r_grid, 30)
        mask = np.zeros(shape[0:2])
        for item in p:
            mask[item] = 1
        return mask
    
    def export_masks(self, a, b, c, shape):        
        print"creating masks..."
        masks = np.zeros(shape)
        
        r_grid = self.create_r_grid(a,b,c)

        for i in range(shape[2]):
            print(float(i)/shape[2])
            
            for j in range(shape[3]):
                p = poisson_disc_light.sample_poisson(shape[0], shape[1], r_grid, 30)
                
                mask = np.zeros(shape[0:2])
                
                for item in p:
                    mask[item]=1
                        
                masks[:,:,i,j] = mask
        
        sp.io.savemat('masks' + '_' + str(a) + '_' + str(b) + '_' + str(c) + '.mat', {'masks':masks})
        
        pass
    
    def import_masks(self, fname):
        masks = sp.io.loadmat(fname)
        masks = masks['masks']
        
        return masks
        
    
    def mask_imgs(self, imgs, fname, method='uniform', p_zeros=0.5):
        print"masking imgs..."
        
        if method == 'poisson':
            try:
                print"importing masks..."
                masks = self.import_masks(fname)
            
            except:
                print"Error while importing masks"
                pass
        
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
    
    def imgs_to_data(self, imgs):
        print"converting imgs to data..."
        data = np.transpose(imgs, (2,0,1,3))
        data = np.reshape(data, (self.timesteps, -1), order='F')
        data = data.T
        return data
    
    def data_to_imgs(self, data):
        print"converting data to imgs..."
        data = data.T
        imgs = np.reshape(data, (self.timesteps, self.cols, self.rows, -1), order='F')
        imgs = np.transpose(imgs,(1,2,0,3))
        return imgs
    
    def imgs_error(self, img1, img2):
        err = np.sqrt(np.sum((img1 - img2) ** 2))
        err /= (img1.shape[0] * img1.shape[1])
        return err
    
    def total_error(self, imgs_ref, imgs_rec):
        err_tot = 0
        for i in range(imgs_ref.shape[3]):
            for j in range(imgs_ref.shape[2]):
                err_tot += self.imgs_error(imgs_ref[:,:,j,i], imgs_rec[:,:,j,i])
        return err_tot
    
    def error_difference(self, imgs, imgs_test, imgs_rec):
        return self.total_error(imgs,imgs_rec) - self.total_error(imgs,imgs_test)
    
    def sparsity(self, A):
        A = np.ravel(A)
        return 1-float(np.count_nonzero(A))/len(A)
    
    def get_training_data(self):
        return
    
    def get_testing_data(self):
        return
    
    def initialize_dictionary(self, n_components, data_train):
        print"initializing dictionary..."
        init = np.zeros((n_components, data_train.shape[1]))
        
        for k in range(int(n_components/2)):
            for n in range(data_train.shape[1]):
                init[k,n] = np.cos((np.pi/data_train.shape[1])*n*(k+0.5))
                
        for i in range(int(n_components/2),n_components):
            init[i,:] = data_train[int(random.uniform(0,data_train.shape[0])),:].copy()
            
        self.init = init
        
        return init

    def train_dictionary(self, imgs_train, n_components=50, alpha=1, transform_alpha=1, 
                         n_iter=250, batch_size=1000, verbose=0, fit_algorithm='lars'):
        '''
        Parameters
        ----------
        method : {'lars', 'cd'}
        lars: uses the least angle regression method to solve the lasso problem
        (linear_model.lars_path)
        cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). Lars will be faster if
        the estimated components are sparse.
        
        alpha : float,
        Sparsity controlling parameter.
        '''
        # normalize images
        imgs_norm = self.normalize(imgs_train)
        # transform to (samples, features) shape
        data_train = self.imgs_to_data(imgs_norm)
        init = self.initialize_dictionary(n_components, data_train)
        
        print"training dictionary..."
        t0 = time.time()
        
        self.dico = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, 
                                                n_iter = n_iter, batch_size=batch_size, 
                                                dict_init=init.copy(), verbose=verbose, 
                                                fit_algorithm='lars')
        self.dico.fit(data_train)
        self.V = self.dico.components_
        
        print"dictionary trained in %.1fs." %(time.time()-t0)
        
        pass
    
    def test_dictionary(self, imgs_test, alpha=1, transform_algorithm='omp', verbose=0):
        '''
        Parameters
        ---------
        alpha : float, 1. by default
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        
        algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}
        lars: uses the least angle regression method (linear_model.lars_path)
        lasso_lars: uses Lars to compute the Lasso solution
        lasso_cd: uses the coordinate descent method to compute the
        Lasso solution (linear_model.Lasso). lasso_lars will be faster if
        the estimated components are sparse.
        omp: uses orthogonal matching pursuit to estimate the sparse solution
        threshold: squashes to zero all coefficients less than alpha from
        the projection dictionary * X'
        '''
        imgs_norm = self.normalize(imgs_test)
        data_test = self.imgs_to_data(imgs_norm)
        
        print"testing dictionary..."
        t0 = time.time()
        
        self.dico.transform_alpha = alpha
        self.dico.transform_algorithm = transform_algorithm
        self.code_test = self.dico.transform(data_test)
        
        recs = np.dot(self.code_test,self.V)
        recs = self.data_to_imgs(recs)
        
        print"dictionary tested in %.1fs." %(time.time()-t0)
        return recs
    
    def minimize_alpha_train(self, imgs_train, imgs_test, imgs_test_ref, 
                             alpha_train=np.linspace(0,1,num=10), n_components=50, 
                             n_iter=250, batch_size=1000, verbose=0, return_vectors=False):
        print"minimizing alpha_train..."
        t0 = time.time()
        err_diff = np.zeros(alpha_train.shape)

        for i in range(len(alpha_train)):

            if verbose > 1:
                print i

            self.train_dictionary(imgs_train, n_components=n_components, 
                                  alpha=alpha_train[i], n_iter=n_iter, 
                                  batch_size=batch_size, verbose=verbose)
            recs = self.test_dictionary(imgs_test)
            err_diff[i] = self.error_difference(obj.normalize(imgs_test_ref), 
                                                obj.normalize(imgs_test), recs)
        
        print"alpha_train minimized in %.1fs" %(time.time()-t0)
        
        min_index = np.argmin(err_diff)
        min_err = err_diff[min_index]
        alpha_min = alpha_train[min_index]
        
        if return_vectors:
            return min_err, alpha_min, err_diff, alpha_train
        
        return min_err, alpha_min
    
    def minimize_alpha_test(self, imgs_train, imgs_test, imgs_test_ref, 
                            alpha_test=np.linspace(0,1,num=10), n_components=50, 
                            n_iter=250, batch_size=1000, verbose=0):
        print"minimizing alpha_test..."
        t0 = time.time()
        err_diff = np.zeros(alpha_test.shape)
        
        for i in range(len(alpha_test)):
            
            if verbose > 1:
                print i
            
            self.train_dictionary(imgs_train, n_components=n_components, 
                                  alpha=alpha_test[i], n_iter=n_iter, batch_size=batch_size, 
                                  verbose=verbose)
            
            recs = self.test_dictionary(imgs_test)
            err_diff[i] = self.error_difference(obj.normalize(imgs_test_ref), 
                                                obj.normalize(imgs_test), recs)
        
        min_index = np.argmin(err_diff)
        min_err = err_diff[min_index]
        
        print"alpha_test minimized in %.1fs" %(time.time()-t0)
        
        return alpha_test, min_err        

#==============================================================================
# main function:
#==============================================================================
if __name__ == '__main__':
    obj = mri_reconstruction()
        
    # Training amount
    percentage = 0.8
    
    # Variables
    imgs_train = obj.imgs[:,:,:,:int(percentage*obj.persons)].copy()
    ref_imgs_test = obj.imgs[:,:,:,int(percentage*obj.persons):obj.persons].copy()
    k_test = obj.transform(ref_imgs_test)
    k_mskd = obj.mask_imgs(k_test, 'masks_200_0_200.mat', method='uniform', p_zeros=0.8)
    imgs_test = obj.inverse_transform(k_mskd)
    
    # Minimize alpha
    #alpha_min, err_min, err_diff, alpha = obj.minimize_alpha_train(imgs_train, imgs_test, ref_imgs_test, 
    #                                                               alpha_train=np.linspace(1,2,num=15), n_components=50, 
    #                                                               n_iter=250, batch_size=1000, verbose=1, return_vectors=True)

    # Training
    obj.train_dictionary(imgs_train, n_components=50, alpha=0.2, n_iter=250, 
                         batch_size=1000, verbose=1, fit_algorithm='lars')
    
    # Testing
    recs = obj.test_dictionary(imgs_test[:,:,:, :2], alpha=0.5, transform_algorithm='lasso_lars')
    
    # calculate error
    #err_diff = obj.error_difference(obj.normalize(ref_imgs_test), obj.normalize(imgs_test), recs)
    obj.sparsity(obj.code_test)
    
    
    def print_imgs():
        print"------------------------------------------------"
        print" "
        
        #printing
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(obj.init, cmap='gray')
        plt.title("Dictionary Initialization")
        plt.subplot(1,3,2)
        plt.imshow(obj.V, cmap='gray')
        plt.title("Dictionary")
        plt.subplot(1,3,3)
        plt.imshow(abs(obj.code_test[0:50,:]), cmap='gray')
        plt.title("code for first 50 Pixels")
        
        plt.figure(figsize=(15,15))
        plt.subplot(2,4,1)
        person1 = 0
        time1 = 0
        plt.imshow(ref_imgs_test[:,:,time1,person1], cmap='gray')
        plt.title("Reference")
        plt.subplot(2,4,2)
        plt.imshow(np.log(abs(k_mskd[:,:,time1,person1])), cmap='gray')
        plt.title("Masked k-space")
        plt.subplot(2,4,3)
        plt.imshow(imgs_test[:,:,time1,person1], cmap='gray')
        plt.title("Test image")
        plt.subplot(2,4,4)
        plt.imshow(recs[:,:,time1,person1], cmap='gray')
        plt.title("Reconstructed image with dictionary")
        
        plt.subplot(2,4,5)
        person2 = 10
        time2 = 0
        plt.imshow(ref_imgs_test[:,:,time2,person2], cmap='gray')
        plt.title("Reference")
        plt.subplot(2,4,6)
        plt.imshow(np.log(abs(k_mskd[:,:,time2,person2])), cmap='gray')
        plt.title("Masked k-space")
        plt.subplot(2,4,7)
        plt.imshow(imgs_test[:,:,time2,person2], cmap='gray')
        plt.title("Test image")
        plt.subplot(2,4,8)
        plt.imshow(recs[:,:,time2,person2], cmap='gray')
        plt.title("Reconstructed image with dictionary")
        
        plt.figure(figsize=(8,8))
        plt.subplot(1,3,1)
        plt.imshow(ref_imgs_test[:,int(obj.cols/2),:,person1],cmap='gray')
        plt.title("Reference")
        plt.subplot(1,3,2)
        plt.imshow(imgs_test[:,int(obj.cols/2),:,person1], cmap='gray')
        plt.title("Inverse transform")
        plt.subplot(1,3,3)
        plt.imshow(recs[:,int(obj.cols/2),:,person1],cmap='gray')
        plt.title("Dictionary Reconstruction")
    
        pass

print_imgs()

