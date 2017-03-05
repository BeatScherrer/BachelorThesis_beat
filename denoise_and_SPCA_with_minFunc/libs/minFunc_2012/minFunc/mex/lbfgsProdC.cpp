#include <math.h>
#include "mex.h"
#include <omp.h>
/* See lbfgsProd.m for details */
/* This function will not exit gracefully on bad input! */

#ifdef USE_WINDOWS
	#define __restrict__ __restrict
#endif

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* Variable Declarations */
	
	//double *S, *Y, *YS, *g, Hdiag, *d, *alpha, *beta;
	double Hdiag;
	double  * __restrict__ S;
	double  * __restrict__ Y;
	double  * __restrict__ YS;
	double  * __restrict__ g;
	double  * __restrict__ d;
	double  * __restrict__ alpha;
	double  * __restrict__ beta;
	
	int nVars,nCor,maxCor,lbfgs_start,lbfgs_end;
	
    #ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs()/2);
    //mexPrintf("%d OMP\n", omp_get_num_procs());
    #endif
    //omp_set_num_threads(6);
	/* Get Input Pointers */
	
	g = mxGetPr(prhs[0]);
	S = mxGetPr(prhs[1]);
	Y = mxGetPr(prhs[2]);
	YS= mxGetPr(prhs[3]);
	lbfgs_start = (int)mxGetScalar(prhs[4]);
	lbfgs_end = (int)mxGetScalar(prhs[5]);
	Hdiag = mxGetScalar(prhs[6]);
	
	if (!mxIsClass(prhs[4],"int32")||!mxIsClass(prhs[5],"int32"))
		mexErrMsgTxt("lbfgs_start and lbfgs_end must be int32");
	
	/* Compute number of variables, maximum number of corrections */
	
	nVars = mxGetDimensions(prhs[1])[0];
	maxCor = mxGetDimensions(prhs[1])[1];
	
	/* Compute number of corrections available */
	if (lbfgs_start == 1)
		nCor = lbfgs_end-lbfgs_start+1;
	else
		nCor = maxCor;
	
	/* Allocate Memory for Local Variables */
	alpha = (double*)mxCalloc(nCor,sizeof(double));
	beta = (double*)mxCalloc(nCor,sizeof(double));
	
	/* Set-up Output Vector */
	plhs[0] = mxCreateDoubleMatrix(nVars,1,mxREAL);
	d = (double*)mxGetPr(plhs[0]);
	
    #pragma omp parallel for schedule(dynamic, 512) 
	for(int j=0;j<nVars;j++) {
		d[j] = -g[j];
	}
	
	for(int i = lbfgs_end-1;i >= 0;i--) {
		alpha[i] = 0;
        double tmp = alpha[i];
        //#pragma omp parallel for default(shared) schedule(dynamic, 512) reduction(+:tmp)
		for(int j=0;j<nVars;j++) {
			tmp = tmp + S[j + nVars*i]*d[j];
		}
		alpha[i] = tmp / YS[i];
        double alpha_i = alpha[i];
        #pragma omp parallel for schedule(dynamic, 512) 
		for(int j=0;j<nVars;j++) {
			d[j] -= alpha_i*Y[j + nVars*i];
		}
	}
	if(lbfgs_start != 1) {
		for(int i = maxCor-1;i >= lbfgs_start-1;i--) {
			alpha[i] = 0;
            double tmp = alpha[i];
            //#pragma omp parallel for default(shared) schedule(dynamic, 512) reduction(+:tmp)
			for(int j=0;j<nVars;j++) {
				tmp = tmp + S[j + nVars*i]*d[j];
			}
			alpha[i] = tmp / YS[i];
            double alpha_i = alpha[i];
            #pragma omp parallel for schedule(dynamic, 512) 
			for(int j=0;j<nVars;j++)
				d[j] -= alpha_i*Y[j + nVars*i];
		}
	}
	
    #pragma omp parallel for schedule(dynamic, 512) 
	for(int j=0;j<nVars;j++) {
		d[j] *= Hdiag;
	}
	
	if(lbfgs_start != 1) {
		for(int i = lbfgs_start-1; i < maxCor; i++) {
			beta[i] = 0;
            double tmp = beta[i];
            //#pragma omp parallel for default(shared) schedule(dynamic, 512) reduction(+:tmp)
			for(int j=0;j<nVars;j++) {
				tmp = tmp + Y[j + nVars*i]*d[j];
			}
			beta[i] = tmp / YS[i];
            double ab_i = alpha[i]-beta[i];
            #pragma omp parallel for schedule(dynamic, 512) 
			for(int j=0;j<nVars;j++) {
				d[j] += S[j+nVars*i]*ab_i;
			}
		}
	}
	for(int i = 0; i < lbfgs_end; i++) {
		beta[i] = 0;
        double tmp = beta[i];
        //#pragma omp parallel for default(shared) schedule(dynamic, 512) reduction(+:tmp)
		for(int j=0;j<nVars;j++) {
			tmp = tmp + Y[j + nVars*i]*d[j];
		}
		beta[i] = tmp / YS[i];
        double ab_i = alpha[i]-beta[i];
        #pragma omp parallel for schedule(dynamic, 512) 
		for(int j=0;j<nVars;j++) {
			d[j] += S[j+nVars*i]*ab_i;
		}
	}
	mxFree(alpha);
	mxFree(beta);
}

/*
 
 void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
     
    double *S, *Y, *YS, *g, Hdiag, *d, *alpha, *beta;
    int i,j,nVars,nCor,maxCor,lbfgs_start,lbfgs_end;
     
     
    g = mxGetPr(prhs[0]);
    S = mxGetPr(prhs[1]);
    Y = mxGetPr(prhs[2]);
    YS= mxGetPr(prhs[3]);
    lbfgs_start = (int)mxGetScalar(prhs[4]);
    lbfgs_end = (int)mxGetScalar(prhs[5]);
    Hdiag = mxGetScalar(prhs[6]);
     
    if (!mxIsClass(prhs[4],"int32")||!mxIsClass(prhs[5],"int32"))
        mexErrMsgTxt("lbfgs_start and lbfgs_end must be int32");
     
     
    nVars = mxGetDimensions(prhs[1])[0];
    maxCor = mxGetDimensions(prhs[1])[1];
     
    if (lbfgs_start == 1)
        nCor = lbfgs_end-lbfgs_start+1;
    else
        nCor = maxCor;
     
    alpha = mxCalloc(nCor,sizeof(double));
    beta = mxCalloc(nCor,sizeof(double));
     
    plhs[0] = mxCreateDoubleMatrix(nVars,1,mxREAL);
    d = mxGetPr(plhs[0]);
     
    for(j=0;j<nVars;j++)
        d[j] = -g[j];
     
    for(i = lbfgs_end-1;i >= 0;i--) {
        alpha[i] = 0;
        for(j=0;j<nVars;j++)
            alpha[i] += S[j + nVars*i]*d[j];
        alpha[i] /= YS[i];
        for(j=0;j<nVars;j++)
            d[j] -= alpha[i]*Y[j + nVars*i];
    }
    if(lbfgs_start != 1) {
        for(i = maxCor-1;i >= lbfgs_start-1;i--) {
            alpha[i] = 0;
            for(j=0;j<nVars;j++)
                alpha[i] += S[j + nVars*i]*d[j];
            alpha[i] /= YS[i];
            for(j=0;j<nVars;j++)
                d[j] -= alpha[i]*Y[j + nVars*i];
        }
    }
     
    for(j=0;j<nVars;j++)
        d[j] *= Hdiag;
     
    if(lbfgs_start != 1) {
        for(i = lbfgs_start-1; i < maxCor; i++) {
            beta[i] = 0;
            for(j=0;j<nVars;j++)
                beta[i] += Y[j + nVars*i]*d[j];
            beta[i] /= YS[i];
            for(j=0;j<nVars;j++)
                d[j] += S[j+nVars*i]*(alpha[i]-beta[i]);
        }
    }
    for(i = 0; i < lbfgs_end; i++) {
        beta[i] = 0;
        for(j=0;j<nVars;j++)
            beta[i] += Y[j + nVars*i]*d[j];
        beta[i] /= YS[i];
        for(j=0;j<nVars;j++)
            d[j] += S[j+nVars*i]*(alpha[i]-beta[i]);
    }
     
    mxFree(alpha);
    mxFree(beta);
     
}
*/
