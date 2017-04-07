# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:50:26 2017

@author: beats
"""

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import poisson_disc

from learn_dictionary import train_dictionary
from learn_dictionary import test_dictionary
from create_mask import create_mask

# import matlab data (johannes')
print("Importing images...")
try:
    imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
    imgs = np.float64(imgs['imgs'])
except:
    print('Error while loading images!')

rows, cols, timesteps, persons = imgs.shape

# Normalizing the images
print("Normalizing the images")
imgs = imgs/(2.0**16)
for i in range(imgs.shape[3]):
    temp = imgs[:,:,:,i].reshape(rows*cols,timesteps)
    temp = temp - np.mean(temp, axis=1, keepdims=True)
    temp = temp / np.std(temp, axis=1, keepdims=True)
    temp[np.isnan(temp)] = 0
    imgs[:,:,:,i] = np.reshape(temp,imgs.shape[:-1])
imgs = np.transpose(imgs, (0,1,2,3))

# Transform to k-Space
print("Transform images to k-space...")
k_imgs = np.fft.fft2(imgs, axes=(0,1))
k_imgs = np.fft.fftshift(k_imgs)

# Mask the k_space data
print("Masking the images...")
mask = create_mask(0.2, (rows,cols), 'gaussian', sigma=20)
k_undersampled = k_imgs.copy()
for i in range(timesteps):
    for j in range(persons):
        k_undersampled[:,:,i,j] = mask * k_imgs[:,:,i,j]

# Reconstruction via ifft2
print("Transform images back from k-space...")
reconstructions_undersampled = np.real(np.fft.ifft2(np.fft.ifftshift(k_undersampled), axes=(0,1)))
reconstructions = np.real(np.fft.ifft2(np.fft.ifftshift(k_imgs), axes=(0,1)))

# Train dictionary on fully sampled data
print("Training...")
train_imgs = imgs[:,:,:,0:int(0.8*persons)]
dico, V = train_dictionary(train_imgs, n_components=50)

# Test dictionary
print("Testing...")
test_imgs = reconstructions_undersampled[:,:,:,int(0.8*persons):persons]
recs = test_dictionary(test_imgs, dico, V)

# Plot various Images
plt.figure
plt.imshow(V,cmap='gray')
plt.title('Dictionary')
plt.figure
plt.subplot(1,5,1)
plt.imshow(imgs[:,:,0,300], cmap='gray')
plt.title('Reference Image')
plt.subplot(1,5,2)
plt.imshow(np.log(np.abs(k_imgs[:,:,0,300])),  cmap='gray')
plt.title('Fully sampled K-space')
plt.subplot(1,5,3)
plt.imshow(np.log(np.abs(k_undersampled[:,:,0,300])),  cmap='gray')
plt.title('Undersampled K-space')
plt.subplot(1,5,4)
plt.imshow(np.real(reconstructions_undersampled[:,:,0,300]),  cmap='gray')
plt.title('Reconstruction of undersampled data')
plt.subplot(1,5,5)
plt.imshow(recs[:,:,0,0], cmap='gray')
plt.title('Reconstruction with Dictionary')
