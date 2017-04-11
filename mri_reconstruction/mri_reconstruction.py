# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:50:26 2017

@author: beats
"""

print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import random

from poisson_disc import Grid
from sklearn.decomposition import MiniBatchDictionaryLearning

from create_mask import create_mask

# import matlab data (johannes')
print("Importing images...")
imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
imgs = np.float64(imgs['imgs'])
rows, cols, timesteps, persons = imgs.shape
s = imgs.shape

# Normalizing the images: mean over time, plus normalize each dimension
print("Normalizing the images...")
imgs = imgs / (2**16)

# normalize data
for i in range(imgs.shape[3]):
  temp = imgs[:,:,:,i].reshape(s[0]*s[1], s[2])
  # remove mean along time???????????
  #temp = temp - np.mean(temp, axis=1, keepdims=True)

  temp = temp - np.median(temp,axis=0,keepdims=True)
  temp = temp / (np.std(temp, axis=0, keepdims=True))
  #temp[np.isnan(temp)] = 0
  imgs[:,:,:,i] = np.reshape(temp, s[:-1])

print("imgs shape:", imgs.shape)


# Transform to k-Space
print("Transform images to k-space...")
k_imgs = np.fft.fft2(imgs, axes=(0,1))
k_imgs = np.fft.fftshift(k_imgs)

# Mask the k_space data 2Do: poisson_disc mask
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

data_train = np.transpose(train_imgs, (2,0,1,3))
#print("data_train", data_train.shape)
data_train = np.reshape(data_train, (timesteps, -1))
data_train = data_train.T
#print("Training Data shape: ", data_train.shape)

# Initialize dictionary as DCT
#init = np.zeros([timesteps,n_components])
#for i in range(n_components):
#    for j in range(timesteps):
#        init[j,i] = np.cos(np.pi/timesteps*(j+0.5)*i)

dico = MiniBatchDictionaryLearning(n_components=50, alpha=0.5, n_iter=500, batch_size=1000, verbose=2) # n_iter = 500, batch_size = 1000
V = dico.fit(data_train).components_

# Test dictionary
print("Testing...")
test_imgs = reconstructions_undersampled[:,:,:,int(0.8*persons):persons]

data_test = np.transpose(test_imgs, (2,0,1,3))
#data_test = np.transpose(np.abs(imgs_under[:,:,:,10], (2, 0, 1))
data_test = np.reshape(data_test, (timesteps, -1))
data_test = data_test.T
code = dico.transform(data_test)
recs = np.dot(code, V)
recs = np.reshape(recs.T, [timesteps, rows, cols, -1])
recs = np.transpose(recs, (1,2,0,3))

code = np.reshape(code.T, [timesteps,rows,cols,-1])
code = np.transpose(code, (1,2,0,3))

# Plot various Images
plt.figure()
plt.imshow(V.T,cmap='gray')
plt.title('Dictionary')
plt.figure(figsize=(40,40))
plt.subplot(2,6,1)
plt.imshow(imgs[:,:,0,300], cmap='gray')
plt.title('Reference Image')
plt.subplot(2,6,2)
plt.imshow(np.log(np.abs(k_imgs[:,:,0,300])),  cmap='gray')
plt.title('Fully sampled K-space')
plt.subplot(2,6,3)
plt.imshow(np.log(np.abs(k_undersampled[:,:,0,300])),  cmap='gray')
plt.title('Undersampled K-space')
plt.subplot(2,6,4)
plt.imshow(np.real(reconstructions_undersampled[:,:,0,300]),  cmap='gray')
plt.title('Reconstruction of undersampled data')
plt.subplot(2,6,5)
plt.imshow(recs[:,:,0,0], cmap='gray')
plt.title('Reconstruction with Dictionary')
plt.subplot(2,6,6)
plt.imshow(code[:,:,0,0],cmap='gray')
plt.title('Sparse Code')

plt.subplot(2,6,7)
plt.imshow(imgs[:,:,0,301], cmap='gray')
plt.title('Reference Image')
plt.subplot(2,6,8)
plt.imshow(np.log(np.abs(k_imgs[:,:,0,301])),  cmap='gray')
plt.title('Fully sampled K-space')
plt.subplot(2,6,9)
plt.imshow(np.log(np.abs(k_undersampled[:,:,0,301])),  cmap='gray')
plt.title('Undersampled K-space')
plt.subplot(2,6,10)
plt.imshow(np.real(reconstructions_undersampled[:,:,0,301]),  cmap='gray')
plt.title('Reconstruction of undersampled data')
plt.subplot(2,6,11)
plt.imshow(recs[:,:,0,1], cmap='gray')
plt.title('Reconstruction with Dictionary')
plt.subplot(2,6,12)
plt.imshow(code[:,:,0,1],cmap='gray')
plt.title('Sparse Code')

print("Dictionary representations are sparse: ", sp.sparse.issparse(code))

def mse(imageA, imageB):
	err = np.sum((imageA - imageB) ** 2)
	err /= (imageA.shape[0] * imageA.shape[1])

	return err

err_tot = 0
for i in range(75):
    for j in range(timesteps):
        err_tot += mse(imgs[:,:,j,300+i],reconstructions_undersampled[:,:,j,300+i])-mse(imgs[:,:,j,300+i], recs[:,:,j,i])

print("Total mse of testing images:", err_tot)