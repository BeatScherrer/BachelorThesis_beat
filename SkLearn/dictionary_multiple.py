# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:16:51 2017

@author: beats
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:51:25 2017

@author: beats
"""


print(__doc__)

from time import time

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image


from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


p = '../Sklearn/images/cartoon/'
img1 = Image.open(p + 'db1.jpg')
img1 = img1.convert('L')
img1 = np.array(img1)

img2 = Image.open(p + 'db2.jpg')
img2 = img2.convert('L')
img2 = np.array(img2)

img3 = Image.open(p + 'db3.jpg')
img3 = img3.convert('L')
img3 = np.array(img3)

images = np.array([img1,img2,img3])

# Convert from uint8 representation with values between 0 and 255 to
# a floating point representation with values between 0 and 1.
images = images / 255
# downsample for higher speed. third argument in array parameter = step size

'''img1 = img1[::2, ::2] + img1[1::2, ::2] + img1[::2, 1::2] + img1[1::2, 1::2]
img1 = img1/4.0
img2 = img2[::2, ::2] + img2[1::2, ::2] + img2[::2, 1::2] + img2[1::2, 1::2]
img2 = img2/4.0
img3 = img3[::2, ::2] + img3[1::2, ::2] + img3[::2, 1::2] + img3[1::2, 1::2]
img3 = img3/4.0'''

height, width = img1.shape

# Extract all reference patches from the whole image of all images
print('Extracting reference patches...')
t0 = time()
patch_size = (5, 5)
data1 = extract_patches_2d(img1, patch_size)
data2 = extract_patches_2d(img2, patch_size)
data3 = extract_patches_2d(img3, patch_size)
data = np.concatenate((data1,data2,data3))
data = data.reshape(data.shape[0], -1)
data = data - np.mean(data, axis=0)
data = data / np.std(data, axis=0)
print('done in %.2fs.' % (time() - t0))

print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=500)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)

# plot the learned dictionary with atoms as patch_size patches
plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Dictionary learned from face patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# construct noisy image to be reconstructed with the above learned dictionary
distorted = img2.copy()
distorted = distorted + 0.075 * np.random.randn(height, width)

print('Extracting noisy patches... ')
t0 = time()
data = extract_patches_2d(distorted, patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))

transform_algorithms = [
    ('Orthogonal Matching Pursuit\n1 atom', 'omp',
     {'transform_n_nonzero_coefs': 1}),
    ('Orthogonal Matching Pursuit\n2 atoms', 'omp',
     {'transform_n_nonzero_coefs': 2}),
    ('Least-angle regression\n5 atoms', 'lars',
     {'transform_n_nonzero_coefs': 5}),
    ('Thresholding\n alpha=0.1', 'threshold', {'transform_alpha': .1})]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + '...')
    reconstructions[title] = img4.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    patches = np.dot(code, V)

    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == 'threshold':
        patches -= patches.min()
        patches /= patches.max()
    reconstructions[title] = reconstruct_from_patches_2d(
        patches, (height, width))
    dt = time() - t0
    print('done in %.2fs.' % dt)