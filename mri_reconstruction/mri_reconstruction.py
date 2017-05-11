import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import random

from sklearn.decomposition import MiniBatchDictionaryLearning

# import matlab data (johannes')
print("Importing images...")
imgs = sp.io.loadmat('ismrm_ssa_imgs.mat')
imgs = np.float64(imgs['imgs'])
rows, cols, timesteps, persons = imgs.shape
s = imgs.shape



# Normalizing the images: mean over time, plus normalize each dimension
print("Normalizing the images...")
imgs = imgs / (2**16)

sample_mean = np.median(imgs, axis = (0, 1), keepdims = True)
sample_std = np.std(imgs, axis = (0, 1), keepdims = True)
sample_std[sample_std < 1e-5] = 1e-5
          
# Transform to k-Space
print("Transform images to k-space...")
k_imgs = np.fft.fft2(imgs, axes=(0,1))
k_imgs = np.fft.fftshift(k_imgs)

# Mask the k_space data
print("Masking the images...")
k_undersampled = k_imgs.copy()
for i in range(persons):
    for j in range(timesteps):
        k_undersampled[:,:,j,i] = k_imgs[:,:,j,i] * mask

# Reconstruction via ifft2
print("Transform images back from k-space...")
reconstructions_undersampled = np.real(np.fft.ifft2(np.fft.ifftshift(k_undersampled), axes=(0,1)))
#reconstructions = np.real(np.fft.ifft2(np.fft.ifftshift(k_imgs), axes=(0,1)))
reconstructions = imgs

imgs = imgs - sample_mean
imgs = imgs / sample_std
reconstructions_undersampled = reconstructions_undersampled - sample_mean
reconstructions_undersampled = reconstructions_undersampled / sample_std

train_mean = np.mean(imgs, axis = 2, keepdims = True)
imgs = imgs - train_mean 

rec_temp_mean = np.mean(reconstructions_undersampled, axis = 2, keepdims = True)
reconstructions_undersampled = reconstructions_undersampled - rec_temp_mean

          
# normalize data
if False:
    for i in range(imgs.shape[3]):
      temp = imgs[:,:,:,i].reshape(s[0]*s[1], s[2])
      
      temp = temp - np.mean(temp, axis=1, keepdims=True)
    
      temp = temp - np.median(temp,axis=0,keepdims=True)
      temp = temp / (np.std(temp, axis=0, keepdims=True))
      temp[np.isnan(temp)] = 0
      imgs[:,:,:,i] = np.reshape(temp, s[:-1])

#print("imgs shape:", imgs.shape)


# Train dictionary on fully sampled data
print("Training...")
train_imgs = imgs[:,:,:,0:int(0.9*persons)]

data_train = np.transpose(train_imgs, (2,0,1,3))
#print("data_train", data_train.shape)
data_train = np.reshape(data_train, (timesteps, -1))
data_train = data_train.T
#print("Training Data shape: ", data_train.shape)

# Initialize dictionary
#init = np.zeros([timesteps,n_components])
#for i in range(n_components):
#    for j in range(timesteps):
#        init[j,i] = np.cos(np.pi/timesteps*(j+0.5)*i)
print("init dict...")
init = np.ndarray((50, timesteps))
for i in range(50):
    init[i,:] = data_train[int(random.uniform(0,data_train.shape[0])),:]

dico = MiniBatchDictionaryLearning(n_components=50, alpha=0.005, n_iter=250, batch_size=1000, dict_init = init, verbose=2) # n_iter = 500, batch_size = 1000
V = dico.fit(data_train).components_

# Test dictionary
print("Testing...")
test_imgs = reconstructions_undersampled[:,:,:,int(0.9*persons):persons]

data_test = np.transpose(test_imgs, (2,0,1,3))
#data_test = np.transpose(np.abs(imgs_under[:,:,:,10], (2, 0, 1))
data_test = np.reshape(data_test, (timesteps, -1))
data_test = data_test.T
code = dico.transform(data_test)
recs = np.dot(code, V)
recs = np.reshape(recs.T, [timesteps, rows, cols, -1])
recs = np.transpose(recs, (1,2,0,3))

# code
code = np.reshape(code.T, [50,rows*cols])

# Plot various Images
plt.figure(figsize = (8,8))
plt.imshow(V.T,cmap='gray')
plt.title('Dictionary')

plt.figure(figsize = (15,15))
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
plt.imshow(code[:,:,0].T,cmap='gray')
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
plt.imshow(code[:,:,1].T,cmap='gray')
plt.title('Sparse Code')

# plot slice over time
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(imgs[:,int(cols/2),:,300],cmap='gray')
plt.subplot(1,2,2)
plt.imshow(recs[:,int(cols/2),:,0],cmap='gray')

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
