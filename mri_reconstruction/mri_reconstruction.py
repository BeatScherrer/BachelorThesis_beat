# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:50:26 2017

@author: beats
"""

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from learn_dictionary import train_dictionary
from learn_dictionary import test_dictionary
from create_mask import create_mask

# import matlab data (johannes')
try:
    imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
    imgs = np.float64(imgs['imgs'])
except:
    print('Error while loading images!')

rows, cols, timesteps, persons = imgs.shape
    
# Transform to k-Space
k_imgs = np.fft.fft2(imgs, axes=(0,1))
k_imgs = np.fft.fftshift(k_imgs)

mask = create_mask(0.3, (rows,cols), 'gaussian', sigma=20)
plt.imshow(mask, cmap='gray')

# Apply mask to all timesteps and persons
k_undersampled = k_imgs.copy()
for i in range(timesteps-1):
    for j in range(persons-1):
        k_undersampled[:,:,i,j] = mask * k_imgs[:,:,i,j]

# Reconstruction via ifft2
reconstructions_undersampled = np.fft.ifft2(np.fft.ifftshift(k_undersampled), axes=(0,1))
reconstructions = np.fft.ifft2(np.fft.ifftshift(k_imgs), axes=(0,1))

# Train dictionary on fully sampled data
dico, V = train_dictionary(imgs, n_components=100)

# Test dictionary
test_img = test_dictionary(reconstructions_undersampled, dico, V)

# Plot various Images
plt.figure
plt.imshow(V,cmap='gray')
plt.title('Dictionary')
plt.figure(figsize=(20,20))
plt.subplot(1,5,1)
plt.imshow(imgs[:,:,1,10], cmap='gray')
plt.title('Reference Image')
plt.subplot(1,5,2)
plt.imshow(np.log(np.abs(k_imgs[:,:,1,10])),  cmap='gray')
plt.title('Fully sampled K-space')
plt.subplot(1,5,3)
plt.imshow(np.log(np.abs(k_undersampled[:,:,1,10])),  cmap='gray')
plt.title('Undersampled K-space')
plt.subplot(1,5,4)
plt.imshow(np.real(reconstructions_undersampled[:,:,1,10]),  cmap='gray')
plt.title('Reconstruction of undersampled data')
plt.subplot(1,5,5)
plt.imshow(test_img[:,:,0], cmap='gray')
plt.title('Reconstruction with Dictionary')
