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

from sklearn.decomposition import MiniBatchDictionaryLearning

from create_mask import create_mask

# import matlab data (johannes')
print("Importing images...")
imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
imgs = np.float64(imgs['imgs'])
rows, cols, timesteps, persons = imgs.shape

# Normalizing the images: mean over time, plus normalize each dimension
print("Normalizing the images...")
imgs = imgs/255
for i in range(imgs.shape[3]):
    temp = imgs[:,:,:,i].reshape(rows*cols,timesteps)
    # remove mean along time
    temp = temp - np.mean(temp, axis=0, keepdims=True)
    #temp = temp -np.median(temp, axis=0, keepdims=True)
    temp = temp / np.std(temp, axis=0, keepdims=True)
    temp[np.isnan(temp)] = 0
    imgs[:,:,:,i] = np.reshape(temp,imgs.shape[:-1])

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

dico = MiniBatchDictionaryLearning(n_components=50, alpha=0.5, n_iter=100, batch_size=3, verbose=2) # n_iter = 500, batch_size = 1000
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

# Plot various Images
plt.figure()
plt.imshow(V,cmap='gray')
plt.title('Dictionary')
plt.figure(figsize=(40,40))
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

print("dictionary representations is sparse: ", sp.sparse.issparse(code))
