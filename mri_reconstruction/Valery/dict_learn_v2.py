from time import time

import numpy as np
import scipy 
import scipy.io

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
import sklearn.feature_extraction

import matplotlib.pyplot as plt

def create_overcomplete_DCT(N, L):
	D = np.zeros((N, L))
	D[:, 0] = 1. / np.sqrt(N)
	xx = np.array(range(N)) * np.pi / L
	for k in range(1, L):
		v = np.cos(xx * k)
		v = v - np.mean(v)
		D[:, k] = v / np.linalg.norm(v)
	return D


def normalize_data_for_learning(data):
	# data : N x N_time
	training_data = data.copy()
	t_mean = np.mean(training_data, axis = 0, keepdims = True)
	training_data = training_data - t_mean
	training_data = training_data - np.mean(training_data, axis = 1, keepdims = True)
	idxs = np.argsort( np.sum(training_data**2, axis = 1) )
	training_data = training_data[idxs[50:], :]
	training_data = training_data / np.std(training_data, axis = 0, keepdims = True)
	# training_data = training_data / np.std(training_data, axis = 1, keepdims = True)
	print 'normalization', np.mean(training_data), np.std(training_data)
	return training_data

def create_matrices(imsz, k_zeros, central_regsz, mask_type, coherent_time):
	mask = []
	rows, cols = imsz[0], imsz[1]
	if mask_type == 'unif':
		if coherent_time == False:
			mask = np.random.rand(*imsz) > k_zeros
		else:
			mask = np.random.rand(imsz[0], imsz[1]) > k_zeros
			mask = np.dstack((mask, ) * imsz[2])
		if central_regsz > 0:
			r1 = int(round(rows/2) - central_regsz)
			r2 = int(round(rows/2) + central_regsz)
			c1 = int(round(cols/2) - central_regsz)
			c2 = int(round(cols/2) + central_regsz)
			mask[r1:r2, c1:c2, :] = 1
	return mask


def create_undersampling(images, masks):
	Nt = images.shape[2]
	Nimgs = images.shape[3]
	masks_shift = masks.copy()
	fimg = np.zeros(images.shape, dtype = np.complex128)
	imgs_zf = np.zeros(images.shape, dtype = np.float64)
	for i in range(Nimgs):
		for t in range(Nt):
			fimg[:, :, t, i] = np.fft.fft2(images[:,:, t, i])
			masks_shift[:, :, t, i] = np.fft.ifftshift(masks[:,:, t, i])
			imgs_zf[:,:, t, i] = np.real(np.abs(np.fft.ifft2(fimg[:,:, t, i] * masks_shift[:,:, t, i])))
	fimg_zf = fimg * masks_shift
	return fimg, masks_shift, fimg_zf, imgs_zf


def Learn_dictionary(imgs, n_components, alpha, fit_algorithm, dict_init, n_iter, batch_size, n_jobs = 1):
	# imgs: rows x cols x N_time x N_images
	N_time = imgs.shape[2]
	training_data = np.transpose(imgs.copy(), (2,0,1,3))
	training_data = np.reshape(training_data, (N_time, -1), order ='F')
	training_data = training_data.T
	training_data = normalize_data_for_learning(training_data)
	dico = MiniBatchDictionaryLearning(alpha = alpha, fit_algorithm = fit_algorithm, \
								    n_iter = n_iter, n_jobs = n_jobs, batch_size = batch_size, shuffle = True, \
									n_components = n_components, verbose = 1, dict_init = dict_init)
	D = dico.fit(training_data).components_
	return D, dico

def Learn_dictionary_spatial(imgs, patch_size, n_components, alpha, fit_algorithm, dict_init, n_iter, batch_size, n_jobs = 1):
	# imgs: rows x cols x N_time x N_images
	N_time = imgs.shape[2]
	N_imgs = imgs.shape[3]
	P = []
	for i in range(N_imgs):
		p = sklearn.feature_extraction.image.extract_patches(imgs[:,:,:, i], [patch_size, patch_size, N_time])
		p = np.reshape(p, [p.shape[0]*p.shape[1], patch_size*patch_size*N_time])
		if i == 0:
			P = p
		else:
			P = np.concatenate((P, p), axis = 0)
		print i

	training_data = normalize_data_for_learning(P)
	print training_data.shape, 'is shape', dict_init.shape
	dico = MiniBatchDictionaryLearning(alpha = alpha, fit_algorithm = fit_algorithm, \
								    n_iter = n_iter, n_jobs = n_jobs, batch_size = batch_size, shuffle = True, \
									n_components = n_components, verbose = 5, dict_init = dict_init)
	D = dico.fit(training_data).components_
	return D, dico


def Recon_BCD(b, masks, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
	n_iter, n_jobs = 1, display = False):
		# b: rows x cols x N_time x N_cases (zero padded kspace), measurements
		# masks: rows x cols x N_time (k-space mask)
		# D  : N_atoms x N_time (dictionary)
		# transform_alpha, transform_n_nonzeros_coefs, transform_algorithm -- sklearn params
	def Recon_v1(b, masks, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
		n_iter, n_jobs = 1, display = False):
		Nt = masks.shape[2]
		rows, cols = masks.shape[0:2]
		Npix = np.float64(rows * cols)
		X = np.zeros((rows, cols, Nt))
		print 'alpha', p_transform_alpha
		coder = SparseCoder(D, transform_algorithm = p_transform_algorithm, transform_alpha = p_transform_alpha, \
							transform_n_nonzero_coefs=p_transform_n_nonzeros_coefs, n_jobs = n_jobs)
		for i in range(Nt):
			X[:,:, i] = np.abs(np.fft.ifft2(b[:,:, i]))
		for it in range(n_iter):
			# encode X
			#U = dico.transform(np.reshape(X, (rows*cols, Nt), order = 'F'))
			U = coder.transform(np.reshape(X, (rows*cols, Nt), order = 'F'))
			Du = np.dot(U, D)
			Du = np.reshape(Du, [rows, cols, Nt], order = 'F')
			# fft force
			for i in range(Nt):
				f = np.fft.fft2(Du[:,:, i])
				f0 = b[:,:,i]
				f[masks[:,:, i]] = f0[masks[:,:, i]]
				X[:,:, i] = np.abs(np.fft.ifft2(f))
			fval = np.sum((X - Du)**2)
			if display:
				print it, ':', fval
		return X, Du, U

	X = np.zeros(b.shape)
	if b.ndim == 3:
		b = b.copy()
		b = b[:,:,:, np.newaxis]
		masks = masks.copy()
		masks = masks[:,:,:, np.newaxis]
		X = X[:,:,:, np.newaxis]
	for i in range(b.shape[3]):
		xtmp, dutmp, utmp = Recon_v1(b[:,:,:,i], masks[:,:,:,i], D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
			n_iter, n_jobs, display)
		X[:, :, :, i] = xtmp
	return X, utmp


def Recon_BCD_spatial(b, masks, patch_size, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
	n_iter, n_jobs = 1, display = False):
		# b: rows x cols x N_time x N_cases (zero padded kspace)
		# masks: rows x cols x N_time (k-space mask)
		# D  : N_atoms x N_time (dictionary)
		# transform_alpha, transform_n_nonzeros_coefs, transform_algorithm -- sklearn params
	def Recon_v1(b, masks, patch_size, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
		n_iter, n_jobs = 1, display = False):
		Nt = masks.shape[2]
		rows, cols = masks.shape[0:2]
		Npix = np.float64(rows * cols)
		X = np.zeros((rows, cols, Nt))
		coder = SparseCoder(D, transform_algorithm = p_transform_algorithm, transform_alpha = p_transform_alpha, \
							transform_n_nonzero_coefs=p_transform_n_nonzeros_coefs, n_jobs = n_jobs)
		for i in range(Nt):
			X[:,:, i] = np.abs(np.fft.ifft2(b[:,:, i]))
		for it in range(n_iter):
			# encode X
			#U = dico.transform(np.reshape(X, (rows*cols, Nt), order = 'F'))
			p = sklearn.feature_extraction.image.extract_patches(X, [patch_size, patch_size, Nt])
			pshp = p.shape
			p = np.reshape(p, [p.shape[0]*p.shape[1], patch_size*patch_size*Nt])
			U = coder.transform(p)
			Du = np.dot(U, D)
			Du = np.reshape(Du, [pshp[0]*pshp[1], patch_size, patch_size, Nt])
			Du = sklearn.feature_extraction.image.reconstruct_from_patches_2d(Du, b.shape)
			# fft force
			for i in range(Nt):
				f0 = b[:,:,i]
				f[masks[:,:, i]] = f0[masks[:,:, i]]
				X[:,:, i] = np.abs(np.fft.ifft2(f))
			fval = np.sum((X - Du)**2)
			if display:
				print it, ':', fval
		return X, Du, U

	X = np.zeros(b.shape)
	if b.ndim == 3:
		b = b.copy()
		b = b[:,:,:, np.newaxis]
		masks = masks.copy()
		masks = masks[:,:,:, np.newaxis]
		X = X[:,:,:, np.newaxis]
	for i in range(b.shape[3]):
		xtmp, dutmp, utmp = Recon_v1(b[:,:,:,i], masks[:,:,:,i], patch_size, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
			n_iter, n_jobs, display)
		X[:, :, :, i] = xtmp
	return X


def R_psnr(xrec, xtrue):
	arg = np.sum((xrec - xtrue)**2 / xrec.size)
	if arg < 1e-15:
		arg = 1e-15
	return np.log10(1. / arg) * 10.

def R_psnr_many(xrec, xtrue):
	psnrs = []
	for i in range(xrec.shape[3]):
		psnrs.append(R_psnr(xrec[:,:,:, i], xtrue[:,:,:, i]))
	return psnrs

def test_spat_DL():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.

	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:10]]
	imgs_test = imgs[:,:,:, perm[374:]]
	patch_size = 5
	n_components = imgs.shape[2] * patch_size * patch_size * 2
	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 5, 'unif', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	V0 = create_overcomplete_DCT(timesteps*patch_size * patch_size, n_components).T
	D1, dico = Learn_dictionary_spatial(imgs_train, patch_size, n_components, 2./4, 'lars', V0.copy(), 50, 1000, n_jobs = 6)
	print 'dict_learned'	
	D = np.concatenate((D1, np.ones((1, timesteps*patch_size * patch_size))/np.sqrt(timesteps*patch_size * patch_size) ))
	Xrec = Recon_BCD_spatial(fimg_zf, masks_shift, patch_size, D, \
			p_transform_alpha = None, p_transform_n_nonzeros_coefs = 4, \
			p_transform_algorithm = 'omp', n_iter = 30, n_jobs = 1, display = True)
	print 'res: = ', np.mean(np.abs(Xrec - imgs_test)**2)

def test_us_size():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	n_components = 60
	
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:300]]
	imgs_test = imgs[:,:,:, perm[373:]]

	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]
	rows, cols, timesteps, n_test = imgs_test.shape
	
	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D1 = V0.copy()
	D1, dico = Learn_dictionary(imgs_train, n_components, 0.5, 'lars', V0.copy(), 150, 1000, n_jobs = 6)
	D = np.concatenate((D1, np.ones((1, timesteps))/np.sqrt(timesteps) ))

	Nus = 10
	k_us = np.linspace(0, 1, Nus)
	res_psnrs = []
	usmp = []
	for i_us in range(Nus):
		for i in range(n_test):
			mm = create_matrices([rows, cols, timesteps], k_us[i_us], 5, 'unif', False)
			if i == 0:
				mask = mm.copy()
				mask = mask[:,:,:, np.newaxis]
			else:
				print mask.shape, mm.shape
				mm = mm[:,:,:, np.newaxis]
				mask = np.concatenate((mask, mm), axis = 3)
		fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)
		scipy.io.savemat('usamp_%d'%i_us, {'fimg':fimg, 'masks_shift':masks_shift, 'img_zf':img_zf, 'imgs_test':imgs_test})
		Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = None, p_transform_n_nonzeros_coefs = 4, \
			p_transform_algorithm = 'omp', n_iter = 30, n_jobs = 1, display = False)
		res_psnrs.append(R_psnr_many(Xrec, imgs_test))
		usmp.append(k_us[i_us])
		print k_us, np.mean(res_psnrs)
		scipy.io.savemat('usamp_test', {'res_psnrs':res_psnrs, 'usmp':usmp, 'Xr':Xrec})

def test_dict_size():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	n_components = 60
	
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:373]]
	imgs_test = imgs[:,:,:, perm[374:]]

	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]
	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.9, 5, 'unif', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			print mask.shape, mm.shape
			mask = np.concatenate((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	res_psnrs = []
	zf_psnrs = []
	ncomps = []
	for iter_ncomp in range(10, 200, 5):
		n_components = iter_ncomp
		V0 = create_overcomplete_DCT(timesteps, n_components).T
		D1 = V0.copy()
		D1, dico = Learn_dictionary(imgs_train, n_components, 1., 'lars', V0.copy(), 100, 1000, n_jobs = 1)
		D = np.concatenate((D1, np.ones((1, timesteps))/np.sqrt(timesteps) ))
		Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = None, p_transform_n_nonzeros_coefs = 3, \
			p_transform_algorithm = 'omp', n_iter = 30, n_jobs = 1, display = False)
		print 'res: ', R_psnr(Xrec, imgs_test), R_psnr(imgs_test, img_zf), iter_ncomp
		res_psnrs.append(R_psnr(Xrec, imgs_test))
		zf_psnrs.append(R_psnr(imgs_test, img_zf))
		ncomps.append(n_components)
		scipy.io.savemat('test_dict_size', {'res_psnr':res_psnrs, 'zf_psnrs':zf_psnrs, 'ncomps' : ncomps})

def test_learning_size():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	n_components = 60
	
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:370]]
	imgs_test = imgs[:,:,:, perm[374:]]

	timesteps = imgs.shape[2]
	#if imgs_test.ndim == 3:
		#imgs_test = imgs_test[:,:,:, np.newaxis]
	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.9, 5, 'unif', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			print mask.shape, mm.shape
			mm = mm[:,:,:, np.newaxis]
			mask = np.concatenate((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	psnr_arr = []
	learn_size = []
	for lrn_sz in range(1, 300, 10):
		V0 = create_overcomplete_DCT(timesteps, n_components).T
		D1 = V0.copy()
		imgs_train = imgs[:,:,:, perm[:lrn_sz]]
		D1, dico = Learn_dictionary(imgs_train, n_components, 1./10, 'lars', V0.copy(), 150, 1000, n_jobs = 1)
		D = np.concatenate((D1, np.ones((1, timesteps))/np.sqrt(timesteps) ))
		Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = None, p_transform_n_nonzeros_coefs = 4, \
			p_transform_algorithm = 'omp', n_iter = 30, n_jobs = 1, display = False)
		plt.imshow(Xrec[:,:, 2, 0]); 
		plt.show()
		print 'res: ', R_psnr(Xrec, imgs_test),
		psnr_arr.append(R_psnr_many(Xrec, imgs_test))
		learn_size.append(lrn_sz)
		#scipy.io.savemat('test_learn_size', {'psnr_arr':psnr_arr, 'learn_size':learn_size})

def test_DL_iters():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.

	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:373]]
	imgs_test = imgs[:,:,:, perm[374:]]
	n_components = 60
	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 5, 'unif', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D1 = V0.copy()
	for dl_it in range(15):
		D1, dico = Learn_dictionary(imgs_train, n_components, 2./4, 'lars', V0.copy(), (dl_it+1)*100, 1000, n_jobs = 1)
		D = np.concatenate((D1, np.ones((1, timesteps))/np.sqrt(timesteps) ))
		Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = None, p_transform_n_nonzeros_coefs = 4, \
			p_transform_algorithm = 'omp', n_iter = 30, n_jobs = 1, display = False)
		print 'res: ', dl_it, '= ', np.mean(np.abs(Xrec - imgs_test)**2)

def test_regul(imgs):
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.

	imgs = imgs.copy()
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:370]]
	imgs_test = imgs[:,:,:, perm[374:]]
	n_components = 60
	timesteps = imgs.shape[2]

	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D, dico = Learn_dictionary(imgs_train, n_components, 2, 'lars', V0.copy(), 100, 1000, n_jobs = 1)
	D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 5, 'unif', True)
		if i == 0:
			mask = mm.copy()
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	for trnz in range(6):
		Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = None, p_transform_n_nonzeros_coefs = trnz + 1, 
			p_transform_algorithm = 'omp', n_iter = 30, n_jobs = 1, display = True)
		print trnz+1, ' res: ', np.mean(np.abs(Xrec - imgs_test)**2)

def test_nnz():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	n_components = 60
	
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:373]]
	imgs_test = imgs[:,:,:, perm[374:]]

	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.9, 5, 'unif', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D1 = V0.copy()
	D1, dico = Learn_dictionary(imgs_train, n_components, 1., 'lars', V0.copy(), 100, 1000, n_jobs = 1)
	D = np.concatenate((D1, np.ones((1, timesteps))/np.sqrt(timesteps) ))
	for nnz in [1, 2, 3, 4, 5, 6, 7]:
		Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = None, p_transform_n_nonzeros_coefs = nnz, \
			p_transform_algorithm = 'omp', n_iter = 30, n_jobs = 1, display = False)
		print 'res: ', R_psnr(Xrec, imgs_test), R_psnr(imgs_test, img_zf), nnz

def run_test():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	rows, cols, timesteps, persons = imgs.shape
	n_components = 60

	# Learn the Dictionary on the extracted patches
	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D, dico = Learn_dictionary(imgs[:,:,:, :55], n_components, 2, 'lars', V0.copy(), 300, 1000, n_jobs = 1)
	D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))


	test_img = np.abs(imgs[:,:,:,300])
	mask = np.random.rand(rows, cols, timesteps) < 0.10
	r1 = int(round(rows/2) - 5)
	r2 = int(round(rows/2) + 5)
	c1 = int(round(cols/2) - 5)
	c2 = int(round(cols/2) + 5)
	mask[r1:r2, c1:c2, :] = 1
	mask_shift = mask.copy()
	fimg = np.zeros((rows, cols, timesteps), dtype=np.complex128)
	img_us_zfill = np.zeros((rows, cols, timesteps), dtype=np.float64)
	for i in range(timesteps):
		fimg[:, :, i] = np.fft.fft2(test_img[:,:, i])
		mask_shift[:, :, i] = np.fft.ifftshift(mask[:,:, i])
		img_us_zfill[:,:, i] = np.abs(np.fft.ifft2( fimg[:,:,i] * mask_shift[:,:,i] ))

	rec = Recon_BCD(fimg*mask_shift, mask_shift, D, None, 5, 'omp', 30, n_jobs = 1, display=True)

	plt.subplot(1,2,1)
	plt.imshow(np.hstack( (img_us_zfill[:,:, 0], rec[:,:, 0], imgs[:,:,0, 300])), interpolation='nearest')
	plt.colorbar()
	plt.subplot(1,2,2)
	plt.imshow(np.concatenate((rec[60, :, :], img_us_zfill[60, :, :], imgs[60, :, :, 300] ))); 
	print np.sum(np.abs(img_us_zfill - imgs[:,:,:,300])), np.sum(np.abs(rec - imgs[:,:,:,300]))
	plt.show()

	plt.subplot(1,3,1)
	plt.imshow(D , interpolation = 'nearest'); plt.colorbar(); 
	plt.subplot(1,3,2)
	plt.imshow(np.matmul(D, D.T) , interpolation = 'nearest'); 
	plt.subplot(1,3,3)
	plt.plot(D.T)
	plt.show()
    

#test_spat_DL()
#test_nnz()
#test_dict_size()
# test_learning_size()
# test_us_size()
if __name__ == '__main__':
    #run_test()
    test_DL_iters()

# 	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
# 	imgs = imgs_s['imgs']
# 	imgs = np.float64(imgs) / 200.
# 	test_regul(imgs)