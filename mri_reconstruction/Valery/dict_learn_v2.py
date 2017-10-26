from time import time

import numpy as np
import scipy 
import scipy.io

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import SparseCoder
import sklearn.feature_extraction

import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit

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
	training_data = training_data / np.std(training_data, axis = 1, keepdims = True)
	print 'normalization:', np.mean(training_data), np.std(training_data)
	return training_data

def create_pois_lines_2d(imsz, k):

	ydim = float(imsz[0])
	rad = np.abs(np.linspace(-1, 1, ydim))
	minval = 0.
	maxval = 1.
	if k != 0:
		R = 1. / k
		frac   = np.floor(1./R*ydim)
		while True:
			val = minval / 2. + maxval / 2.
			pdf = (1.-rad)**R + val
			pdf[pdf > 1] = 1.
			tot = np.floor(np.sum(pdf))
			if tot > frac:
				maxval = val
			if tot < frac:
				minval = val
			if tot == frac:
				break
		pdf[pdf > 1] = 1.
		pdfsum = np.sum(pdf)
		tmp = np.zeros(pdf.shape)
		while np.abs(np.sum(tmp) - pdfsum) > 1.:
			tmp = np.random.rand(*pdf.shape) < pdf
	else:
		tmp = np.zeros(pdf.shape)
	tmp[int(np.ceil(ydim / 2.))] = 1
	tmp[int(np.floor(ydim / 2.))] = 1
	mask = np.stack( (tmp,)*imsz[1], 1 )
	return mask


def create_matrices(imsz, k_zeros, central_regsz, mask_type, coherent_time):
	imsz = np.array(imsz)
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
	if mask_type == 'poisson_lines':
		if coherent_time == False:
			if imsz.ndim == 2:
				imsz = [imsz[0], imsz[1], 1]
			masks = np.zeros(imsz)
			for i in range(imsz[2]):
				masks[:,:, i] = create_pois_lines_2d( (imsz[0], imsz[1]), 1 - k_zeros)
		
	return masks


def create_undersampling(images, masks):
	if images.ndim < 4:
		images = np.expand_dims(images, axis=4)
 		masks = np.expand_dims(masks, axis=4)       
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
	return fimg, masks, masks_shift, fimg_zf, imgs_zf


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

def soft_thresh(x, l):
    return x * np.maximum(np.absolute(x) - l, 0.) / np.maximum(np.absolute(x), l)

def proximal_l1(x, lam):
	return np.sign(x) * np.maximum(np.abs(x) - lam, 0.)


def Recon_ADMM(b, masks, D, p_transform_alpha, n_iter, n_jobs = 1, display = False, step = 0.001):
		# b: rows x cols x N_time x N_cases (zero padded kspace)
		# masks: rows x cols x N_time (k-space mask)
		# D  : N_atoms x N_time (dictionary)
		# transform_alpha, transform_n_nonzeros_coefs, transform_algorithm -- sklearn params
	def Recon_ADMM_v1(b, masks, D, p_transform_alpha, n_iter, n_jobs = 1, display = False, step = 0.001):
		Nt = masks.shape[2]
		Nat = D.shape[1]
		rows, cols = masks.shape[0:2]
		Npix = (rows * cols)

		rho_x = 10.
		rho_t = 10.
		rho_z = 10.

		# rho_x = 1110.01
		# rho_t = 1110.01
		# rho_z = 1110.01

		ap_eps = np.sqrt(p_transform_alpha)

		X = np.zeros((Nt, Npix))
		U = np.zeros((Nat, Npix))
		Z = np.zeros((Nat, Npix))
		T = np.zeros((Nt, Npix))
		
		Fz = np.zeros((Nat, Npix))
		Ft = np.zeros((Nt, Npix))
		Fx = np.zeros((Nt, Npix), dtype = np.complex128)

		fval = 0

		Du = np.matmul(D, U)
		Dtmp = rho_t * np.matmul(D.T, D) + rho_z * np.eye(Nat)
		Dc, Dl = scipy.linalg.cho_factor(Dtmp)

		for i in range(Nt):
			X[i, :] = np.abs(np.fft.ifft2(b[:,:, i])).ravel(order = 'F')

		# step = 1.0
		fvals = []
		zvals = []
		xvals = []
		tvals = []
		for it in range(n_iter):
			# encode X
			X_old = X.copy()
			for i in range(Nt):
				a1 = b[:,:, i] - np.reshape(Fx[i, :], (rows, cols), order = 'F')
				a2 = Du[i, :] + T[i, :] - Ft[i, :]
				a2 = np.reshape(a2, (rows, cols), order = 'F')
				tmp = np.real(np.fft.ifft2( a1 * masks[:,:, i].squeeze() ))
				rhs = rho_x * tmp + rho_t * a2
				tmp = np.fft.fft2(rhs) / (rho_x * masks[:,:, i].astype(np.float64).squeeze() + rho_t)
				tmp = np.real(np.fft.ifft2(tmp)).astype(np.float64)
				X[i, :] = tmp.ravel(order = 'F')

			Fxtmmp = np.reshape(Fx.T, (rows, cols, Nt), order = 'F')
			lossX = lambda X: rho_t/2*np.sum((X - Du - T + Ft)**2) + rho_x/2 * np.sum(np.real(masks * np.fft.fft2(np.reshape(X.T, (rows, cols, Nt), order='F') ,axes=(0,1))- b + Fxtmmp)**2)
			#print 'XXX, ou', lossX(X), lossX(X_old), lossX(X) - lossX(X_old)
			# Z
			Z_old = Z.copy()
			Z = proximal_l1(U + Fz, 1./rho_z)
			
			# debug
			fz = lambda Z : np.sum(np.abs(Z)) + rho_z/2. * np.sum( ((U+Fz) - Z)**2)
			#print 'ZZZ ou', fz(Z_old), fz(Z), fz(Z) - fz(Z_old)

			# U
			U_old = U.copy()
			Du_old = Du.copy()
			rhs = rho_t * np.matmul(D.T, X - T + Ft) + rho_z * (Z - Fz)
			U = scipy.linalg.cho_solve((Dc, Dl), rhs)
			Du = np.matmul(D, U)
			lossu = lambda U, Du: rho_z/2 * np.sum((U - Z + Fz)**2) + rho_t/2 * np.sum((X - T + Ft - Du)**2)
			#print 'UUU ou', lossu(U_old, Du_old), lossu(U, Du), lossu(U, Du) - lossu(U_old, Du_old)

			# T
			A = X - Du + Ft
			nrm = np.sqrt(np.sum(A**2))
			if nrm > ap_eps:
				A = A / nrm * ap_eps
			T = A.copy()
			#print 'TTT ou', np.sqrt(np.sum(A**2)) - ap_eps

			# fvals.append(np.sum(np.sum(Z)))
			# zvals.append(np.sum( (U - Z)**2 ))
			# tvals.append(np.sum( (X - Du - T)**2 ))
			# xvals.append(np.sum( ( masks * np.fft.fft2(np.reshape(X.T, (rows, cols, Nt), order = 'F'), axes=(0,1)) - b )**2 ))

			Fz += U - Z
			Ft += X - Du - T
			for i in range(Nt):
				tmp = np.fft.fft2(np.reshape(X[i, :], (rows, cols), order = 'F')) * masks[:,:, i]
				Fx[i, :] += (tmp - b[:,:,i]).ravel(order = 'F')

			fval = np.sum(np.abs(Z))
			fvalu = np.sum((U-Z)**2)
			if display:
				print it, ':', fval, fvalu
			#if dx < 1e-6:
			#	break

		# plt.plot(fvals)
		# plt.show()
		# plt.plot(zvals)
		# plt.show()
		# plt.plot(tvals)
		# plt.show()
		# plt.plot(xvals)
		# plt.show()
		# plt.plot(Z.ravel())
		# plt.show()
		# plt.plot(np.sort(Z.ravel()))
		# plt.show()
		return X, Du, U

	X = np.zeros(b.shape)
	if b.ndim == 3:
		b = b.copy()
		b = b[:,:,:, np.newaxis]
		masks = masks.copy()
		masks = masks[:,:,:, np.newaxis]
		X = X[:,:,:, np.newaxis]
	for i in range(b.shape[3]):
		xtmp, dutmp, utmp = Recon_ADMM_v1(b[:,:,:,i], masks[:,:,:,i].astype(np.bool), D, p_transform_alpha, n_iter, n_jobs, display, step)
		xtmp = np.transpose(xtmp)
		xtmp = np.reshape(xtmp, b[:,:,:,i].shape, order = 'F')
		X[:, :, :, i] = xtmp
	return X


def Recon_IST(b, masks, D, p_transform_alpha, n_iter, n_jobs = 1, display = False, step = 0.001):
		# b: rows x cols x N_time x N_cases (zero padded kspace)
		# masks: rows x cols x N_time (k-space mask)
		# D  : N_atoms x N_time (dictionary)
		# transform_alpha, transform_n_nonzeros_coefs, transform_algorithm -- sklearn params
	def Recon_IST_v1(b, masks, D, p_transform_alpha, n_iter, n_jobs = 1, display = False, step = 0.001):
		Nt = masks.shape[2]
		rows, cols = masks.shape[0:2]
		Npix = np.float64(rows * cols)
		X = np.zeros((rows, cols, Nt))
		U = np.zeros((rows*cols, D.shape[0]))
		fval = 0

		for i in range(Nt):
			X[:,:, i] = np.abs(np.fft.ifft2(b[:,:, i]))
		# step = 1.0
		for it in range(n_iter):
			# encode X
			#U = dico.transform(np.reshape(X, (rows*cols, Nt), order = 'F'))
			X_p = X.copy()
			U_p = U.copy()
			fval_p = fval

			dtmp = np.reshape(X, (rows*cols, Nt), order = 'F')
			dtmpm = np.mean(dtmp, axis = 1, keepdims = True)
			
			# if (it + 1) % 10 == 0:
			# 	step *= 1.4
			
			Du = np.dot(U, D)
			grad = np.matmul(D, (Du - dtmp).T ).T 
			U = soft_thresh(U - grad / step, p_transform_alpha / step)
			
			Du = np.dot(U, D)
			resid = dtmp - dtmpm - Du
			Du = Du + dtmpm
			Du = np.reshape(Du, [rows, cols, Nt], order = 'F')
			# fft force
			for i in range(Nt):
				f = np.fft.fft2(Du[:,:, i])
				f0 = b[:,:,i]
				f[masks[:,:, i]] = f0[masks[:,:, i]]
				X[:,:, i] = np.abs(np.fft.ifft2(f))
			fval = np.sum((X - Du)**2) + p_transform_alpha * np.sum(np.abs(U))
			dx = np.max((X_p - X)**2)

			if it > 0 and fval > fval_p:
				X = X_p.copy()
				U = U_p.copy()
				fval = fval_p
				step = step * 2.

			if display:
				print it, ':', fval, dx, step
			if dx < 1e-6:
				break
		return X, Du, U

	X = np.zeros(b.shape)
	if b.ndim == 3:
		b = b.copy()
		b = b[:,:,:, np.newaxis]
		masks = masks.copy()
		masks = masks[:,:,:, np.newaxis]
		X = X[:,:,:, np.newaxis]
	for i in range(b.shape[3]):
		xtmp, dutmp, utmp = Recon_IST_v1(b[:,:,:,i], masks[:,:,:,i].astype(np.bool), D, p_transform_alpha, n_iter, n_jobs, display, step)
		X[:, :, :, i] = xtmp
	return X


def Recon_BCD(b, masks, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
	n_iter, n_jobs = 1, display = False):
		# b: rows x cols x N_time x N_cases (zero padded kspace)
		# masks: rows x cols x N_time (k-space mask)
		# D  : N_atoms x N_time (dictionary)
		# transform_alpha, transform_n_nonzeros_coefs, transform_algorithm -- sklearn params
	def Recon_v1(b, masks, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
		n_iter, n_jobs = 1, display = False):
		Nt = masks.shape[2]
		rows, cols = masks.shape[0:2]
		Npix = np.float64(rows * cols)
		X = np.zeros((rows, cols, Nt))
		if p_transform_algorithm == 'omp' and p_transform_alpha:
			omp = OrthogonalMatchingPursuit(tol = p_transform_alpha)
			print 'My omp'
		else:
			coder = SparseCoder(dictionary = D.copy(), transform_algorithm = p_transform_algorithm, transform_alpha = p_transform_alpha, \
							transform_n_nonzero_coefs=p_transform_n_nonzeros_coefs, n_jobs = n_jobs)

		for i in range(Nt):
			X[:,:, i] = np.abs(np.fft.ifft2(b[:,:, i]))
		for it in range(n_iter):
			# encode X
			#U = dico.transform(np.reshape(X, (rows*cols, Nt), order = 'F'))
			dtmp = np.reshape(X, (rows*cols, Nt), order = 'F')
			dtmpm = np.mean(dtmp, axis = 1, keepdims = True)
			
			if p_transform_algorithm == 'omp' and p_transform_alpha:
				U = omp.fit(D.T, (dtmp - dtmpm).T).coef_
			else:
				U = coder.transform(dtmp - dtmpm)

			Du = np.dot(U, D)
			resid = dtmp - dtmpm - Du
			Du = Du + dtmpm
			Du = np.reshape(Du, [rows, cols, Nt], order = 'F')
			# fft force
			X_prev = X.copy()
			for i in range(Nt):
				f = np.fft.fft2(Du[:,:, i])
				f0 = b[:,:,i]
				f[masks[:,:, i]] = f0[masks[:,:, i]]
				X[:,:, i] = np.abs(np.fft.ifft2(f))
			fval = np.sum((X - Du)**2)
			dx = np.max((X_prev - X)**2)
			if display:
				print it, ':', fval, dx
			if dx < 1e-6:
				break
		return X, Du, U

	X = np.zeros(b.shape)
	if b.ndim == 3:
		b = b.copy()
		b = b[:,:,:, np.newaxis]
		masks = masks.copy()
		masks = masks[:,:,:, np.newaxis]
		X = X[:,:,:, np.newaxis]
	for i in range(b.shape[3]):
		xtmp, dutmp, utmp = Recon_v1(b[:,:,:,i], masks[:,:,:,i].astype(np.bool), D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
			n_iter, n_jobs, display)
		X[:, :, :, i] = xtmp
	return X




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
			X_prev = X.copy()
			for i in range(Nt):
				f = np.fft.fft2(Du[:,:, i])
				f0 = b[:,:,i]
				f[masks[:,:, i]] = f0[masks[:,:, i]]
				X[:,:, i] = np.abs(np.fft.ifft2(f))
			fval = np.sum((X - Du)**2)
			dx = np.max((X_prev - X)**2)
			if display:
				print it, ':', fval, dx
			if dx < 1e-3:
				break
		return X, Du, U

	X = np.zeros(b.shape)
	if b.ndim == 3:
		b = b.copy()
		b = b[:,:,:, np.newaxis]
		masks = masks.copy()
		masks = masks[:,:,:, np.newaxis]
		X = X[:,:,:, np.newaxis]
	for i in range(b.shape[3]):
		xtmp, dutmp, utmp = Recon_v1(b[:,:,:,i], masks[:,:,:,i].astype(np.bool), patch_size, D, p_transform_alpha, p_transform_n_nonzeros_coefs, p_transform_algorithm, \
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
	imgs_test = imgs[:,:,:, perm[370:]]

	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]
	rows, cols, timesteps, n_test = imgs_test.shape
	
	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D1 = V0.copy()
	D1, dico = Learn_dictionary(imgs_train, n_components, 0.5, 'lars', V0.copy(), 150, 1000, n_jobs = 6)
	D = np.concatenate((D1, np.ones((1, timesteps))/np.sqrt(timesteps) ))

	Nus = 15
	k_us = np.linspace(0.0001, 0.999, Nus)
	res_psnrs = []
	usmp = []
	for i_us in range(Nus):
		for i in range(n_test):
			mm = create_matrices([rows, cols, timesteps], k_us[i_us], 5, 'poisson_lines', False)
			if i == 0:
				mask = mm.copy()
				mask = mask[:,:,:, np.newaxis]
			else:
				mm = mm[:,:,:, np.newaxis]
				mask = np.concatenate((mask, mm), axis = 3)
		fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)
		scipy.io.savemat('usamp_%d'%i_us, {'fimg':fimg, 'masks_shift':masks_shift, 'img_zf':img_zf, 'imgs_test':imgs_test})
		Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = 0.1, p_transform_n_nonzeros_coefs = None, \
			p_transform_algorithm = 'lasso_cd', n_iter = 30, n_jobs = 1, display = True)
		res_psnrs.append(R_psnr_many(Xrec, imgs_test))
		#usmp.append(k_us[i_us])
		usmp.append(np.mean(masks_shift))
		print k_us[i_us], np.mean(R_psnr_many(Xrec, imgs_test))
		scipy.io.savemat('usamp_test_2', {'res_psnrs':res_psnrs, 'usmp':usmp, 'Xr':Xrec})

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
		mm = create_matrices([rows, cols, timesteps], 0.9, 5, 'poisson_lines', False)
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

def test_regulIST():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.

	imgs = imgs.copy()
	perm = np.random.permutation(imgs.shape[3])
	#imgs_train = imgs[:,:,:, perm[:370]]
	#imgs_test = imgs[:,:,:, perm[374:]]
	imgs_train = imgs[:,:,:, :370]
	imgs_test = imgs[:,:,:, 374:]
	n_components = 50
	timesteps = imgs.shape[2]

	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D, dico = Learn_dictionary(imgs_train, n_components, 2., 'lars', V0.copy(), 100, 1000, n_jobs = 6)
	D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'poisson_lines', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	alphas = np.linspace(0, 0.0005, 35)
	psnr_arr = []
	for trnz in range(alphas.size):
		print 'TTT', trnz
		alpha = alphas[trnz]
		Xrec = Recon_IST(fimg_zf, masks_shift, D, \
			p_transform_alpha = alpha, n_iter = 200, n_jobs = 1, display = True, step = 1000.)
		psnr_arr.append(R_psnr_many(Xrec, imgs_test))
		scipy.io.savemat('test_alphaIST2', {'psnr_arr':psnr_arr, 'alphas':alphas})


def test_regul_forDL():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.

	imgs = imgs.copy()
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, :370]
	imgs_test = imgs[:,:,:, 374:]
	n_components = 60
	timesteps = imgs.shape[2]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'poisson_lines', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	alphas = np.linspace(1e-10, 5., 60) 
	#(0.002 - 1e-10)/2 * 128*128*25
	psnr_arr = []
	for trnz in range(alphas.size):
		alpha = alphas[trnz]
		alpha_rec = (0.002 - 1e-10)/2 * 128*128*25
		print alpha
		V0 = create_overcomplete_DCT(timesteps, n_components).T
		D, dico = Learn_dictionary(imgs_train, n_components, alpha, 'lars', V0.copy(), 300, 1000, n_jobs = 6)
		D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))
		Xrec = Recon_ADMM(fimg_zf, masks_shift, D.T, \
			p_transform_alpha = alpha_rec, n_iter = 100, n_jobs = 1, display = False)
		psnr_arr.append(R_psnr_many(Xrec, imgs_test))
		scipy.io.savemat('test_alpha_DL_Wstd_lars', {'psnr_arr':psnr_arr, 'alphas':alphas})

def test_Dsize():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.

	imgs = imgs.copy()
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, :370]
	imgs_test = imgs[:,:,:, 374:]
	n_components = 60
	timesteps = imgs.shape[2]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'poisson_lines', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	comps_arr = range(5, 150, 5)
	psnr_arr = []
	for icomp in range(len(comps_arr)):
		n_components = comps_arr[icomp]
		alpha_rec = (0.002 - 1e-10)/2 * 128*128*25
		print n_components
		V0 = create_overcomplete_DCT(timesteps, n_components).T
		D, dico = Learn_dictionary(imgs_train, n_components, 1.2, 'cd', V0.copy(), 500, 1000, n_jobs = 6)
		D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))
		Xrec = Recon_ADMM(fimg_zf, masks_shift, D.T, \
			p_transform_alpha = alpha_rec, n_iter = 100, n_jobs = 1, display = False)
		psnr_arr.append(R_psnr_many(Xrec, imgs_test))
		scipy.io.savemat('test_dsize', {'psnr_arr':psnr_arr, 'comps_arr':comps_arr})

def test_LRNsize():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.

	imgs = imgs.copy()
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, :370]
	imgs_test = imgs[:,:,:, 374:]
	n_components = 60
	timesteps = imgs.shape[2]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'poisson_lines', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	lrn_arr = range(1, 100, 5)
	psnr_arr = []
	for ilrn in range(len(lrn_arr)):
		lrn_sz = lrn_arr[ilrn]
		imgs = imgs.copy()
		imgs_train = imgs[:,:,:, :lrn_sz]
		imgs_test = imgs[:,:,:, 374:]
		n_components = 60
		timesteps = imgs.shape[2]

		rows, cols, timesteps, n_test = imgs_test.shape
		for i in range(n_test):
			mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'poisson_lines', False)
			if i == 0:
				mask = mm.copy()
				mask = mask[:,:,:, np.newaxis]
			else:
				mask = np.stack((mask, mm), axis = 3)
		fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)
		alpha_rec = (0.002 - 1e-10)/2 * 128*128*25
		print lrn_sz
		V0 = create_overcomplete_DCT(timesteps, n_components).T
		D, dico = Learn_dictionary(imgs_train, n_components, 1.2, 'cd', V0.copy(), 1000, 1000, n_jobs = 6)
		D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))
		Xrec = Recon_ADMM(fimg_zf, masks_shift, D.T, \
			p_transform_alpha = alpha_rec, n_iter = 100, n_jobs = 1, display = False)
		psnr_arr.append(R_psnr_many(Xrec, imgs_test))
		scipy.io.savemat('test_lrnsize', {'psnr_arr':psnr_arr, 'lrn_arr':lrn_arr})


def test_regul():
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
	D, dico = Learn_dictionary(imgs_train, n_components, 2., 'lars', V0.copy(), 100, 1000, n_jobs = 6)
	D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'poisson_lines', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	alphas = np.linspace(1e-10, 0.002, 50) * 128*128*25
	#(0.002 - 1e-10)/2 * 128*128*25
	psnr_arr = []
	for trnz in range(alphas.size):
		alpha = alphas[trnz]
		alg = 'lasso_lars'
		alg = 'lars'
		alg = 'omp'
		# Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
		# 	p_transform_alpha = alpha, p_transform_n_nonzeros_coefs = None, 
		# 	p_transform_algorithm = alg, n_iter = 100, n_jobs = 1, display = True)
		print alpha
		Xrec = Recon_ADMM(fimg_zf, masks_shift, D.T, \
			p_transform_alpha = alpha, n_iter = 100, n_jobs = 1, display = False)
		psnr_arr.append(R_psnr_many(Xrec, imgs_test))
		scipy.io.savemat('test_alpha', {'psnr_arr':psnr_arr, 'alphas':alphas})

def test_nnz():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	n_components = 60
	
	perm = np.random.permutation(imgs.shape[3])
	imgs_train = imgs[:,:,:, perm[:370]]
	imgs_test = imgs[:,:,:, perm[374:]]
	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'unif', False)
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


def run_test_usual():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	n_components = 60
	
	imgs_train = imgs[:,:,:, :370]
	imgs_test = imgs[:,:,:, 374:]

	timesteps = imgs.shape[2]
	if imgs_test.ndim == 3:
		imgs_test = imgs_test[:,:,:, np.newaxis]

	rows, cols, timesteps, n_test = imgs_test.shape
	for i in range(n_test):
		mm = create_matrices([rows, cols, timesteps], 0.8, 2, 'poisson_lines', False)
		if i == 0:
			mask = mm.copy()
			mask = mask[:,:,:, np.newaxis]
		else:
			mask = np.stack((mask, mm), axis = 3)
	fimg, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D1 = V0.copy()
	D1, dico = Learn_dictionary(imgs_train, n_components, 1.2, 'cd', V0.copy(), 3000, 1000, n_jobs = 1)
	D = np.concatenate((D1, np.ones((1, timesteps))/np.sqrt(timesteps) ))

	Xrec = Recon_BCD(fimg_zf, masks_shift, D, \
			p_transform_alpha = 0.1, p_transform_n_nonzeros_coefs = None, \
			p_transform_algorithm = 'lasso_cd', n_iter = 100, n_jobs = 1, display = False)
	#Xrec2 = Recon_IST(fimg_zf, masks_shift, D, p_transform_alpha = 0.1, n_iter = 300, n_jobs = 1, display = False, step = 1000.)
	Xrec2 = Recon_ADMM(fimg_zf, masks_shift, D.T, p_transform_alpha = (0.002 - 1e-10)/2 * 128*128*25, n_iter = 300, n_jobs = 1, display = False, step = 1000.)
	print 'res: ', R_psnr(Xrec, imgs_test), R_psnr(Xrec2, imgs_test), R_psnr(imgs_test, img_zf)

	plt.subplot(1,3,1)
	plt.imshow(np.hstack( (img_zf[:,:, 0, 0], imgs_test[:,:, 0, 0], Xrec[:,:,0, 0], Xrec2[:,:,0, 0])), interpolation='nearest')
	plt.subplot(1,3,2)
	plt.imshow(np.hstack( (img_zf[:,60, :, 0], imgs_test[:,60, :, 0], Xrec[:,60,:, 0], Xrec2[:,60, :, 0])), interpolation='nearest')

	plt.subplot(1,3,3)
	plt.imshow(D , interpolation = 'nearest'); 
	plt.show()


def run_test():
	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	imgs = imgs_s['imgs']
	imgs = np.float64(imgs) / 200.
	rows, cols, timesteps, persons = imgs.shape
	n_components = 60
	alpha_rec = 210

	# Learn the Dictionary on the extracted patches
	V0 = create_overcomplete_DCT(timesteps, n_components).T
	D, dico = Learn_dictionary(imgs[:,:,:, :55], n_components, 1.0, 'lars', V0.copy(), 300, 1000, n_jobs = 1)
	D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))


	imgs_test = np.abs(imgs[:,:,:,300])
	masks = create_matrices(imgs_test.shape, 0.9, 2, 'poisson_lines', False)

	fimg, masks, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, masks)
	x = Recon_ADMM(fimg_zf, masks_shift, D.T, alpha_rec, n_iter=100, n_jobs = 1, display=True)
	rec = x
    
	print"\n fimg:", fimg.shape, "masks_shift:", masks_shift.shape, "fimg_zf:", fimg_zf.shape, "img_zf:", img_zf.shape, "rec:", rec.shape

	plt.imshow(V0, cmap='gray')
    
	plt.imshow(masks[:,:,0,0], cmap='gray')
	plt.title('Zero-filling mask')
    
	plt.figure(figsize=(15,15))
	plt.subplot(1,4,1)
	plt.imshow(imgs_test[:,:,0], cmap='gray')
	plt.title('Reference')
	plt.subplot(1,4,2)
	plt.imshow(img_zf[:,:,0,0], cmap='gray')
	plt.title('Undersampled Test Image')
	plt.subplot(1,4,3)
	plt.imshow(rec[:,:,0,0], cmap='gray')
	plt.title('Reconstruction')
	plt.subplot(1,4,4)
	plt.imshow((imgs_test[:,:,0]-rec[:,:,0,0])**2, cmap='gist_heat')
	plt.title('Squared Error Map')
    
	plt.figure(figsize=(10,10))
	plt.subplot(1,4,1)
	plt.imshow(imgs_test[int(imgs_test.shape[0]/2),:,:], cmap='gray')
	plt.title('Reference')
	plt.xlabel('Frame')
	plt.ylabel('x-Axis')
	plt.subplot(1,4,2)
	plt.imshow(img_zf[int(img_zf.shape[0]/2),:,:,0], cmap='gray')
	plt.title('Undersampled test image')
	plt.xlabel('Frame')
	plt.ylabel('x-Axis')
	plt.subplot(1,4,3)
	plt.imshow(rec[int(rec.shape[0]/2),:,:,0], cmap='gray')
	plt.title('Reconstruction')
	plt.xlabel('Frame')
	plt.ylabel('x-Axis')
	plt.subplot(1,4,4)
	plt.imshow(((imgs_test-rec[:,:,:,0])**2)[int(imgs_test.shape[0]/2),:,:], cmap='gist_heat')
	plt.title('MSE Map')
	plt.xlabel('Frame')
	plt.ylabel('x-Axis')
	plt.tight_layout(h_pad=1.2)
    
	plt.figure(figsize=(10,10))
	plt.plot(imgs_test[int(imgs_test.shape[1]/2),:,0,0], label='reference', color='b')
	plt.plot(img_zf[int(img_zf.shape[1]/2),:,0,0], label='test', color='y')
	plt.plot(rec[int(rec.shape[1]/2),:,0], label='rec', color='c')
	plt.plot((imgs_test[int(imgs_test.shape[1]/2),:,0,0]-rec[int(rec.shape[1]/2),:,0])**2,label='MSE', color='r')
	plt.legend()
	plt.xlabel('x')
	plt.ylabel('Intensity')
    
if __name__ == '__main__':
	#test_DL_iters()
	#test_spat_DL()
	#test_nnz()
	#test_dict_size()
	# test_learning_size()
	#test_us_size()
	#run_test_usual()
	#test_regul()
	#test_regul_forDL()
#	run_test_usual()
	# test_Dsize()
	#test_LRNsize()
#	run_test()
	# if __name__ == '__main__':
#	 	run_test()

	# 	imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
	# 	imgs = imgs_s['imgs']
	# 	imgs = np.float64(imgs) / 200.
	# 	test_regul(imgs)
    
    imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
    imgs = imgs_s['imgs']
    imgs = np.float64(imgs) / 200.
    rows, cols, timesteps, persons = imgs.shape
    n_components = 60
    alpha_rec = 210

    # Learn the Dictionary on the extracted patches
    V0 = create_overcomplete_DCT(timesteps, n_components).T
    D, dico = Learn_dictionary(imgs[:,:,:, :55], n_components, 1.0, 'lars', V0.copy(), 300, 1000, n_jobs = 1)
    D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))


    imgs_test = np.abs(imgs[:,:,:,300])
    masks = create_matrices(imgs_test.shape, 0.9, 2, 'poisson_lines', False)

    fimg, masks, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, masks)
    x = Recon_ADMM(fimg_zf, masks_shift, D.T, alpha_rec, n_iter=100, n_jobs = 1, display=True)
    rec = x

    def make_plots():
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.imshow(V0, cmap='gray')
        plt.title('Dictionary initialization')
        plt.xlabel('features')
        plt.ylabel('atoms')
        plt.subplot(1,2,2)
        plt.imshow(D, cmap='gray')
        plt.title('Dictionary learned on training data')
        plt.xlabel('features')
        plt.ylabel('atoms')

        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1)
        plt.imshow(masks[:,:,0,0], cmap='gray')
        plt.title('Zero-filling mask')
        plt.subplot(1,3,2)
        plt.imshow(abs(np.log(np.fft.fftshift((np.fft.fft2(imgs_test[:,:,0]))))),cmap='gray')
        plt.title('Fully sampled k-space data')
        plt.subplot(1,3,3)
        plt.imshow(abs(np.log(np.fft.fftshift((np.fft.fft2(imgs_test[:,:,0]))))*masks[:,:,0,0]),cmap='gray')
        plt.title('Zero-filled k-space data')
        
    
        plt.figure(figsize=(15,15))
        plt.subplot(1,4,1)
        plt.imshow(imgs_test[:,:,0], cmap='gray')
        plt.title('Reference')
        plt.subplot(1,4,2)
        plt.imshow(img_zf[:,:,0,0], cmap='gray')
        plt.title('Undersampled Test Image')
        plt.subplot(1,4,3)
        plt.imshow(rec[:,:,0,0], cmap='gray')
        plt.title('Reconstruction')
        plt.subplot(1,4,4)
        plt.imshow((imgs_test[:,:,0]-rec[:,:,0,0])**2, cmap='gist_heat')
        plt.title('Squared Error Map')
    	
        plt.figure(figsize=(10,10))
        plt.subplot(1,4,1)
        plt.imshow(imgs_test[int(imgs_test.shape[0]/2),:,:], cmap='gray')
        plt.title('Reference')
        plt.xlabel('Frame')
        plt.ylabel('x-Axis')
        plt.subplot(1,4,2)
        plt.imshow(img_zf[int(img_zf.shape[0]/2),:,:,0], cmap='gray')
        plt.title('Undersampled test image')
        plt.xlabel('Frame')
        plt.ylabel('x-Axis')
        plt.subplot(1,4,3)
        plt.imshow(rec[int(rec.shape[0]/2),:,:,0], cmap='gray')
        plt.title('Reconstruction')
        plt.xlabel('Frame')
        plt.ylabel('x-Axis')
        plt.subplot(1,4,4)
        plt.imshow(((imgs_test-rec[:,:,:,0])**2)[int(imgs_test.shape[0]/2),:,:], cmap='gist_heat')
        plt.title('MSE Map')
        plt.xlabel('Frame')
        plt.ylabel('x-Axis')
        plt.tight_layout(h_pad=1.2)
    
        plt.figure(figsize=(10,10))
        plt.plot(imgs_test[int(imgs_test.shape[1]/2),:,0], label='reference', color='b')
        plt.plot(img_zf[int(img_zf.shape[1]/2),:,0,0], label='test', color='y')
        plt.plot(rec[int(rec.shape[1]/2),:,0], label='rec', color='c')
        plt.plot((imgs_test[int(imgs_test.shape[1]/2),:,0]-rec[int(rec.shape[1]/2),:,0,0])**2,label='MSE', color='r')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Intensity')
    make_plots()