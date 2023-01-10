/*=========================================================
 * matrixDivide.c - Example for illustrating how to use 
 * LAPACK within a C MEX-file.
 *
 * X = matrixDivide(A,B) computes the solution to a 
 * system of linear equations A * X = B
 * using LAPACK routine DGESV, where 
 * A is a real N-by-N matrix.
 * X and B are real N-by-1 matrices.
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2009-2017 The MathWorks, Inc.
 *=======================================================*/

#if !defined(_WIN32)
#define dgesv dgesv_
#endif

#include "mex.h"
#include "lapack.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *A, *B;    /* pointers to input matrices */
    double *A2, *B2;  /* in/out arguments to DGESV*/
    size_t m,n,p;     /* matrix dimensions */ 
    mwSignedIndex *iPivot;   /* inputs to DGESV */
    mxArray *Awork, *mxPivot;
    mwSignedIndex info, dims[2];

 	/* Check for proper number of arguments. */
    if ( nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:matrixDivide:rhs",
            "This function requires 2 input matrices.");
    }

    A = mxGetPr(prhs[0]); /* pointer to first input matrix */
    B = mxGetPr(prhs[1]); /* pointer to second input matrix */
    /* dimensions of input matrices */
    m = mxGetM(prhs[0]);  
    p = mxGetN(prhs[0]);
    n = mxGetN(prhs[1]);

    /* Validate input arguments */
    if (p != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt("MATLAB:matrixDivide:matchdims",
            "Inner dimensions of matrices do not match.");
    }
    if (p != m) {
        mexErrMsgIdAndTxt("MATLAB:matrixDivide:square",
            "LAPACK function requires input matrix 1 must be square.");
    }
    if (n != 1) {
        mexErrMsgIdAndTxt("MATLAB:matrixDivide:zerodivide",
            "For this example input matrix 2 must be a column vector.");  
    }

    /* check to make sure the first input argument is a real matrix */
    if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
      mexErrMsgIdAndTxt( "MATLAB:matrixDivide:fieldNotRealMatrix",
              "First input argument must be a real matrix.");
    }
    /* check to make sure the second input argument is a real matrix */
    if( !mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
      mexErrMsgIdAndTxt( "MATLAB:matrixDivide:fieldNotRealMatrix",
              "Second input argument must be a real matrix.");
    }

    /* DGESV works in-place, so we copy the inputs first. */
    Awork = mxCreateDoubleMatrix(m, p, mxREAL);
    A2 = mxGetPr(Awork);
    plhs[0] = mxCreateDoubleMatrix(p, n, mxREAL);
    B2 = mxGetPr(plhs[0]);
    memcpy(A2, A, m*p*mxGetElementSize(prhs[0]));
    memcpy(B2, B, p*n*mxGetElementSize(prhs[1]));

    /* Create inputs for DGESV */
    dims[0] = m;
    dims[1] = p;
    mxPivot = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    iPivot = (mwSignedIndex*)mxGetData(mxPivot);

    /* Call LAPACK */
    dgesv(&m,&n,A2,&m,iPivot,B2,&p,&info);
    /* plhs[0] now holds X */

    mxDestroyArray(Awork);
    mxDestroyArray(mxPivot);
}
