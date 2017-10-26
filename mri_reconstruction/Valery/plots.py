# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:26:33 2017

@author: beats
"""

import numpy as np
import scipy 
import scipy.io
import os

import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

mat_ADMM = scipy.io.loadmat('us_test/admm.mat')
mat = scipy.io.loadmat('us_test/usamp.mat')
mat_KT = scipy.io.loadmat('us_test/ktFOCUSS.mat')

def make_plots(m, idx, line):
    
#        plt.figure(figsize=(10,10))
#        plt.subplot(1,2,1)
#        plt.imshow(V0, cmap='gray')
#        plt.title('Dictionary initialization')
#        plt.xlabel('features')
#        plt.ylabel('atoms')
#        plt.subplot(1,2,2)
#        plt.imshow(D, cmap='gray')
#        plt.title('Dictionary learned on training data')
#        plt.xlabel('features')
#        plt.ylabel('atoms')
#    
#        plt.figure(figsize=(10,10))
#        plt.subplot(1,3,1)
#        plt.imshow(np.fft.ifftshift(mat['masks_shift'][:,:,0,m,idx],axes=(0,1)), cmap='gray')
#        plt.title('Zero-filling mask')
#        plt.subplot(1,3,2)
#        plt.imshow(abs(np.log(np.fft.fftshift((np.fft.fft2(mat['imgs_test'][:,:,0,m,idx])),axes=(0,1)))),cmap='gray')
#        plt.title('Fully sampled k-space data')
#        plt.subplot(1,3,3)
#        plt.imshow(abs(np.log(np.fft.ifftshift(np.fft.fft2(mat['imgs_test'][:,:,0,m,idx])*mat['masks_shift'][:,:,0,m,idx]))),cmap='gray')
#        plt.title('Zero-filled k-space data')
    
    # Spatial comparison kt-focus and admm
    se_kt_spat = (mat['imgs_test'][:,:,10,m,idx]-abs(mat_KT['xtrec'][:,:,10,m,idx]))**2
    se_admm_spat = (mat['imgs_test'][:,:,10,m,idx]-abs(mat_ADMM['Xrec'][:,:,10,m,idx]))**2
    kt_spat_max = max(np.ravel(((mat['imgs_test'][:,:,10,m,idx]-abs(mat_KT['xtrec'][:,:,10,m,idx]))**2)))
    admm_spat_max = max(np.ravel(((mat['imgs_test'][:,:,10,m,idx]-abs(mat_ADMM['Xrec'][:,:,10,m,idx]))**2)))
    
    if kt_spat_max > admm_spat_max:
        peak_spat = kt_spat_max
    else:
        peak_spat = admm_spat_max
    
    print peak_spat
    
    print"Undersampling factor:", 1-np.count_nonzero(np.ravel(mat['masks_shift'][:,:,10,m,idx]))/(128.0**2)
    plt.figure(figsize=(15,15))
    plt.subplot(3,2,1)
    plt.imshow(mat['imgs_test'][:,:,10,m,idx],cmap='gray')
    plt.plot(np.arange(0,128,1), np.zeros(128)+line, 'r--')
    plt.title('Fully sampled reference')
    plt.axis('off')
    plt.subplot(3,2,2)
    plt.imshow(mat['img_zf'][:,:,10,m,idx],cmap='gray')
    plt.plot(np.arange(0,128,1), np.zeros(128)+line, 'r--')
    plt.title('Undersampled test image')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(3,2,3)
    plt.imshow(abs(mat_KT['xtrec'][:,:,10,m,idx]),cmap='gray')
    plt.plot(np.arange(0,128,1), np.zeros(128)+line, 'r--')
    plt.title('kt-FOCUSS reconstruction')
    plt.axis('off')
    plt.subplot(3,2,4)
    plt.imshow(se_kt_spat, cmap='gist_heat')
    plt.plot(np.arange(0,128,1), np.zeros(128)+line, 'r--')
    plt.title('MSE kt-FOCUSS')
    plt.colorbar()
    plt.clim(0,peak_spat)
    plt.axis('off')
    plt.subplot(3,2,5)
    plt.imshow(abs(mat_ADMM['Xrec'][:,:,10,m,idx]),cmap='gray')
    plt.plot(np.arange(0,128,1), np.zeros(128)+line, 'r--')
    plt.title('ADMM reconstruction')
    plt.axis('off')
    plt.subplot(3,2,6)
    plt.imshow(se_admm_spat, cmap='gist_heat')
    plt.plot(np.arange(0,128,1), np.zeros(128)+line, 'r--')
    plt.title('MSE ADMM')
    plt.colorbar()
    plt.clim(0,peak_spat)
    plt.axis('off')
    plt.subplots_adjust(hspace=0.1, wspace=-0.55)
    
    # Temporal images
    se_kt_temp = (((mat['imgs_test'][:,:,:,m,idx]-abs(mat_KT['xtrec'][:,:,:,m,idx]))**2)[line,:,:]).T
    se_admm_temp = (((mat['imgs_test'][:,:,:,m,idx]-abs(mat_ADMM['Xrec'][:,:,:,m,idx]))**2)[line,:]).T
    kt_temp_max = max(np.ravel(se_kt_temp))
    admm_temp_max = max(np.ravel(se_admm_temp))
    
    if kt_temp_max > admm_temp_max:
        peak_temp = kt_temp_max
    else:
        peak_temp = admm_temp_max
    
    plt.figure(figsize=(15,15))
    plt.subplot(6,1,1)
    plt.imshow((mat['imgs_test'][line,:,:,m,idx]).T, cmap='gray')
    plt.title('Reference')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(6,1,2)
    plt.imshow((mat['img_zf'][line,:,:,m,idx]).T, cmap='gray')
    plt.title('Undersampled test image')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(6,1,3)
    plt.imshow((abs(mat_KT['xtrec'][line,:,:,m,idx])).T, cmap='gray')
    plt.title('kt-FOCUSS reconstruction')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(6,1,4)
    plt.imshow(se_kt_temp, cmap='gist_heat')
    plt.title('MSE kt-FOCUSS')
    plt.colorbar()
    plt.clim(0,peak_temp)
    plt.axis('off')
    plt.subplot(6,1,5)
    plt.imshow((abs(mat_ADMM['Xrec'][line,:,:,m,idx])).T, cmap='gray')
    plt.title('ADMM reconstruction')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(6,1,6)
    plt.imshow(se_admm_temp, cmap='gist_heat')
    plt.title('MSE ADMM')
    plt.colorbar()
    plt.clim(0,peak_temp)
    plt.axis('off')
    plt.subplots_adjust(hspace=0.2)
    
    # undersampling
#        plt.figure(figsize=(15,15))
#        plt.subplot(2,3,1)
#        plt.imshow(mat['imgs_test'][:,:,10,m,idx],cmap='gray')
#        plt.title('Fully sampled reference')
#        plt.axis('off')
#        plt.subplot(2,3,2)
#        plt.imshow(np.fft.ifftshift(mat['masks_shift'][:,:,10,m,idx],axes=(0,1)), cmap='gray')
#        plt.title('Zero-filling mask $M_1$ at $t$')
#        plt.axis('off')
#        plt.subplot(2,3,3)
#        plt.imshow(mat['img_zf'][:,:,10,m,idx],cmap='gray')
#        plt.title('Zero-filled image')
#        plt.axis('off')
#        plt.subplot(2,3,5)
#        plt.imshow(np.fft.ifftshift(mat['masks_shift'][:,:,11,m,idx],axes=(0,1)), cmap='gray')
#        plt.title('Zero-filling mask $M_2$ at $t+1$')
#        plt.axis('off')
#        plt.subplot(2,3,6)
#        plt.imshow(abs(np.fft.ifft2(np.fft.fft2(mat['imgs_test'][:,:,10,m,idx])*mat['masks_shift'][:,:,12,m,idx])),cmap='gray')
#        plt.title('Zero-filled image')
#        plt.axis('off')
#        plt.subplots_adjust(wspace=0.1,hspace=-0.4)
    
    # Temporal slice
    plt.figure(figsize=(10,10))
    plt.plot(mat['imgs_test'][int(mat['imgs_test'].shape[1]/2),:,0,m,idx], label='reference', color='b')
    plt.plot(mat['img_zf'][int(mat['img_zf'].shape[1]/2),:,0,m,idx], label='test', color='y')
    plt.plot(mat_ADMM['Xrec'][int(mat_ADMM['Xrec'].shape[1]/2),:,0,m,idx], label='rec ADMM', color='c')
    plt.plot(mat_KT['xtrec'][int(mat_KT['xtrec'].shape[1]/2),:,0,m,idx], label='rec KT-FOCUSS', color='c')
    plt.plot((mat['imgs_test'][int(mat['imgs_test'].shape[1]/2),:,0,m,idx]-mat_ADMM['Xrec'][int(mat_ADMM['Xrec'].shape[1]/2),:,0,m,idx])**2,label='MSE ADMM', color='r')
    plt.plot((mat['imgs_test'][int(mat['imgs_test'].shape[1]/2),:,0,m,idx]-mat_KT['xtrec'][int(mat_KT['xtrec'].shape[1]/2),:,0,m,idx])**2,label='MSE KT-FOCUSS', color='r')
    plt.legend()
    plt.title('Intensity plot along the red line')
    plt.ylabel('Intensity')
    
    plt.figure(figsize=(20,20))
    plt.subplot(1,4,1)
    plt.imshow(mat['imgs_test'][:,:,0,0,18],cmap='gray')
#    plt.title('Undersampling mask for undersampling factor of 0.8')
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(np.log(np.fft.fftshift(abs(mat['fimg'])[:,:,0,0,18])),cmap='gray')
#    plt.title('Undersampling mask for undersampling factor of 0.91')
    plt.axis('off')
    plt.subplot(1,4,3)
    plt.imshow(np.log(np.fft.fftshift(abs(mat['masks_shift'][:,:,0,0,18])*abs(mat['fimg'][:,:,0,0,18]))),cmap='gray')
#    plt.title('Undersampling mask for undersampling factor of 0.95')
    plt.axis('off')
    plt.subplot(1,4,4)
    plt.imshow(mat['img_zf'][:,:,0,0,18], cmap='gray')
    plt.axis('off')
    
def gif(filename, array, fps=10, scale=2.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images: (frames, dim1, dim2)
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'
    
    array = np.transpose(array, (2,0,1))
    array = abs(array)
    array *= 255/(max(array.ravel()))
#    plt.imshow(array[15,:,:],cmap='gray')
    
    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip
    
    
# m:3 -> line:52
# m:2 -> line:70 (nice dynamics)
#m=3
#us = [19,22,23]
#line = 52
#for idx in us:
#    make_plots(3,idx,line)
#    
#plt.figure(figsize=(15,15))
#plt.imshow(V0.T,cmap='gray')
#plt.xlabel('atoms')
#plt.ylabel('features')
#plt.savefig('test.svg')