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
    """ Creates overcomplete DCT dictionary
    Args:
        N: vector size
        L: dictionary size / number of atoms
    Returns:
        Dictionary D os (NxL) size
    """
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
    print 'normalization', np.mean(training_data), np.std(training_data)
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


def Recon_ADMM(b, masks, D, p_transform_alpha, n_iter, display = False, tol_x = 0.006):
    """ ADMM reconstruction
        min |U|_1
        s.t. Mask_i * FFT * x_i = b_i, for all i
             |X - DU|_2 < p_transform_alpha
        note that this formulation has global tolerance on encoding
    
    Args:
        b: rows x cols x N_time x N_cases (zero padded kspace)
        masks: rows x cols x N_time (k-space mask)
        D  : N_time x N_atoms (dictionary)
        p_transform_alpha -- epsilon (see prob definition)
        n_iter : max number of iterations
        n_jobs : don't use it
        display : print info on every iteration
        tol_x : termination criteria. Terminates if max(X - X_previous) <= tol_x
    """
    def Recon_ADMM_v1(b, masks, D, p_transform_alpha, n_iter, n_jobs = 1, display = False, step = 0.001, \
                    rho_x = 10., rho_t = 10., rho_z = 10., tol_x = 0.006, debug = False, log_history = False):
        Nt = masks.shape[2]
        Nat = D.shape[1]
        rows, cols = masks.shape[0:2]
        Npix = (rows * cols)
        ap_eps = np.sqrt(p_transform_alpha)
        X = np.zeros((Nt, Npix))
        U = np.zeros((Nat, Npix))
        Z = np.zeros((Nat, Npix))
        T = np.zeros((Nt, Npix))
        Fz = np.zeros((Nat, Npix))
        Ft = np.zeros((Nt, Npix))
        Fx = np.zeros((Nt, Npix), dtype = np.complex128)

        Du = np.matmul(D, U)
        Dtmp = rho_t * np.matmul(D.T, D) + rho_z * np.eye(Nat)
        Dc, Dl = scipy.linalg.cho_factor(Dtmp)

        for i in range(Nt):
            X[i, :] = np.abs(np.fft.ifft2(b[:,:, i])).ravel(order = 'F')
        if log_history:
            Xhist, fvals, zvals, xvals, tvals = [], [], [], [], []

        fval = 0
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
            if debug:
                Fxtmmp = np.reshape(Fx.T, (rows, cols, Nt), order = 'F')
                lossX = lambda X: rho_t/2*np.sum((X - Du - T + Ft)**2) + rho_x/2 * np.sum(np.real(masks * np.fft.fft2(np.reshape(X.T, (rows, cols, Nt), order='F') ,axes=(0,1))- b + Fxtmmp)**2)
                print 'XXX ', lossX(X), lossX(X_old), lossX(X) - lossX(X_old)
            # Z
            Z_old = Z.copy()
            Z = proximal_l1(U + Fz, 1./rho_z)
            if debug:
                fz = lambda Z : np.sum(np.abs(Z)) + rho_z/2. * np.sum( ((U+Fz) - Z)**2)
                print 'ZZZ ', fz(Z_old), fz(Z), fz(Z) - fz(Z_old)
            # U
            U_old = U.copy()
            Du_old = Du.copy()
            rhs = rho_t * np.matmul(D.T, X - T + Ft) + rho_z * (Z - Fz)
            U = scipy.linalg.cho_solve((Dc, Dl), rhs)
            Du = np.matmul(D, U)
            if debug:
                lossu = lambda U, Du: rho_z/2 * np.sum((U - Z + Fz)**2) + rho_t/2 * np.sum((X - T + Ft - Du)**2)
                print 'UUU ', lossu(U_old, Du_old), lossu(U, Du), lossu(U, Du) - lossu(U_old, Du_old)
            # T
            A = X - Du + Ft
            nrm = np.sqrt(np.sum(A**2))
            if nrm > ap_eps:
                A = A / nrm * ap_eps
            T = A.copy()
            if debug:
                print 'TTT ', np.sqrt(np.sum(A**2)) - ap_eps

            if log_history:
                fvals.append(np.sum(np.abs(Z)))
                zvals.append(np.sum( (U - Z)**2 ))
                tvals.append(np.sum( (X - Du - T)**2 ))
                xvals.append(np.sum( ( masks * np.fft.fft2(np.reshape(X.T, (rows, cols, Nt), order = 'F'), axes=(0,1)) - b )**2 ))
                Xhist.append(np.reshape(np.transpose(X.copy()), b[:,:,:].shape, order = 'F'))
            # dual updates
            Fz += U - Z
            Ft += X - Du - T
            for i in range(Nt):
                tmp = np.fft.fft2(np.reshape(X[i, :], (rows, cols), order = 'F')) * masks[:,:, i]
                Fx[i, :] += (tmp - b[:,:,i]).ravel(order = 'F')

            fval = np.sum(np.abs(Z))
            fvalu = np.sum((U-Z)**2)
            if display:
                print it, ':', fval, fvalu, 'max', np.max(np.abs(X - X_old))
            if np.max(np.abs(X - X_old)) < 0.006:
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
        xtmp, dutmp, utmp = Recon_ADMM_v1(b[:,:,:,i], masks[:,:,:,i].astype(np.bool), D, p_transform_alpha, n_iter, display, tol_x)
        xtmp = np.transpose(xtmp)
        xtmp = np.reshape(xtmp, b[:,:,:,i].shape, order = 'F')
        X[:, :, :, i] = xtmp
    return X, utmp


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
                print it, ':', fval, dx, np.sum(np.abs(U))
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


if __name__ == '__main__':
    Nsmpls = 1
    imgs_s = scipy.io.loadmat('data_J/ismrm_ssa_imgs.mat')
    imgs = imgs_s['imgs']
    np.random.seed(6)
    img_s = None
    imgs = np.float64(imgs) / 200.
    recon_alpha = 208.
    Nlearn = 100
    Ntest = 1
    us_factor = 0.8
    n_components = 60

    perm = np.random.permutation(imgs.shape[3])
    print 'perm', perm[-Ntest:]
    imgs_train = imgs[:,:,:, perm[:Nlearn]]
    imgs_test = imgs[:,:,:, perm[-Ntest:]]
    timesteps = imgs.shape[2]

    V0 = create_overcomplete_DCT(timesteps, n_components).T
    D, dico = Learn_dictionary(imgs_train, n_components, 1.2, 'cd', V0.copy(), 5000, 1000, n_jobs = 1)
    D = np.concatenate((D, np.ones((1, timesteps))/np.sqrt(timesteps) ))
    scipy.io.savemat('temporal_dict_1.2_cd_60_100lrn_5000_1000_DCTinit.mat', {'D': D})
    rows, cols, timesteps, n_test = imgs_test.shape
    for i in range(n_test):
        mm = create_matrices([rows, cols, timesteps], us_factor, 2, 'poisson_lines', False)
        if i == 0:
            mask = mm.copy()
            mask = mask[:,:,:, np.newaxis]
        else:
            mm = mm[:,:,:, np.newaxis]
            mask = np.concatenate((mask, mm), axis = 3)
    fimg, masks, masks_shift, fimg_zf, img_zf = create_undersampling(imgs_test, mask)

    Xrec, utmp = Recon_ADMM(fimg_zf, masks_shift, D.T, \
            p_transform_alpha = recon_alpha, n_iter = 100, display = True)
    # XrecBC = Recon_BCD(fimg_zf, masks_shift, D, p_transform_alpha = recon_alpha, \
    #     p_transform_n_nonzeros_coefs = [], p_transform_algorithm = 'omp', n_iter = 20, n_jobs = 1, display = True)
    XrecBC = Recon_BCD(fimg_zf, masks_shift, D, p_transform_alpha = 208. / fimg_zf.size, \
        p_transform_n_nonzeros_coefs = [], p_transform_algorithm = 'omp', n_iter = 50, n_jobs = 1, display = True)
    scipy.io.savemat('res_tst_.mat', {'xA':Xrec, 'xB':XrecBC, 'xT': imgs_test, 'masks': mask})