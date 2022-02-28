##########################################################################
#                            Tank Priyansu
#                            Python | 2022
#                      priyansutank3044@gmail.com
#
# Func sortAlong is used for svDecomp
# Func foTrans is used independently
# Func crossSpectral needs hannWindow and dofo which is also a sub-func of foTrans
##########################################################################

import numpy as np
from numpy.core.numeric import cross
from scipy import linalg as LA
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

def sortAlong(imat, smat):
    smat = smat.T
    mat = np.column_stack((smat, imat))

    mat = mat[np.flip(mat[:, 0].argsort())]

    return np.array(mat[:, 1:])

def svDecomp(mat):
    aat = mat @ mat.T
    ata = mat.T @ mat

    if (mat.shape[0]>mat.shape[1]):
        eig = LA.eig(ata)
        eigaat = LA.eig(aat)

        eigvals = eig[0]
        eigvals = np.flip(eigvals[eigvals[:].argsort()])
        eigvals = np.diag(eigvals)
        s1 = np.sqrt(eigvals)
        s2 = np.zeros((mat.shape[0]-mat.shape[1], mat.shape[1]))
        sing = np.vstack((s1, s2))

        U = sortAlong(eigaat[1].T, eigaat[0]).T

        V = sortAlong(eig[1].T, eig[0]).T

    elif (mat.shape[0]<mat.shape[1]):
        eig = LA.eig(aat)
        eigata = LA.eig(ata)
        
        eigvals = eig[0]
        eigvals = np.flip(eigvals[eigvals[:].argsort()])
        eigvals = np.diag(eigvals)
        s1 = np.sqrt(eigvals)
        s2 = np.zeros((mat.shape[0], mat.shape[1]-mat.shape[0]))
        sing = np.hstack((s1, s2))

        U = sortAlong(eig[1].T, eig[0]).T

        V = sortAlong(eigata[1].T, eigata[0]).T        
    else:
        eig = LA.eig(aat)
        eigata = LA.eig(ata)

        eigvals = eig[0]
        eigvals = np.flip(eigvals[eigvals[:].argsort()])
        eigvals = np.diag(eigvals)
        sing = np.sqrt(eigvals)

        U = sortAlong(eig[1].T, eig[0]).T

        V = sortAlong(eigata[1].T, eigata[0]).T
    
    return U, sing, V

def foTrans(x, s):
    ##########################################################################
    # Input Parameters:
    # x: 1D array of which fft needs to be calculated
    # s: sampling rate of x signal
    # 
    # Output Parameter:
    # X: 1D array containing fft(x) coefficients
    # freq: 1D array containing frequency corresponding to each coeff in x
    ##########################################################################

    N = len(x)
    n = np.arange(N)
    T = N/s             #Time
    freq = n/T          #Freq

    # Actual FFT function based on Cooley-Tukey's Algorithm
    def dofo(x):
        N = len(x)
        if N == 1:
            return x
        else:
            X0 = dofo(x[::2])
            X1 = dofo(x[1::2])
            omega = np.exp((-2j*np.pi*np.arange(N))/N)

            X = np.concatenate([X0 + omega[:int(N/2)]*X1, X0 + omega[int(N/2):]*X1])
        return X

    X = dofo(x)

    return X, freq

def dofo(x):
        N = len(x)
        if N == 1:
            return x
        else:
            X0 = dofo(x[::2])
            X1 = dofo(x[1::2])
            omega = np.exp((-2j*np.pi*np.arange(N))/N)

            X = np.concatenate([X0 + omega[:int(N/2)]*X1, X0 + omega[int(N/2):]*X1])
        return X

def hannWindow(no):
    n = np.arange(0, no)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * n / (no - 1))
    return w

def crossSpectral(x, y, fs):
    ##########################################################################
    # Input Parameters
    # x & y are input signals for which csd is computed, not necessarily distinct
    # fs is the sampling rate
    #
    # Output Parameters:
    # fn = Sample Frequencies
    # csd = Cross & Auto Correlation based on if input x & y are distinct or same
    #
    # Default Parameters:
    # Window: Hanning of length 256
    # Overlap: 50% 
    # These default parameters are set so that by default, fft algorithm needs to compute 
    # fft of segments having only length 256
    #
    # Note: It clips signal from end in a way that total length becomes multiple of 256, 
    # doesn't affect result much 
    #
    # L: Window Length
    # fn = Sample Frequencies
    # D = Step
    # K = Number of Total Signal Segments
    #
    # Based on:
    # The use of Fast Fourier Transformfor the Estimation of Power Spectra: A Method
    # Based on Time Averageing Over Short, Modified Periodograms, Peter D. Welch, 1967
    ##########################################################################

    if len(x) > 256:
        x = x[:len(x)-(len(x)%256)]
        y = y[:len(x)-(len(x)%256)]
    x = signal.detrend(x, axis=0)
    y = signal.detrend(y, axis=0)

    L = 256

    fn = np.arange(L//2+1)*fs/L

    window = hannWindow(L)

    D = 128
    K = int(1 + (len(x) - L)//D)

    start = 0
    end = start + L - 1

    dataSeg1 = np.zeros((K, L), dtype=complex)
    dataSeg2 = np.zeros((K, L), dtype=complex)
    fft1 = np.zeros((K, L), dtype=complex)
    fft2 = np.zeros((K, L), dtype=complex)

    for i in range(K):
        dataSeg1[i,:] = x[int(start):int(end)+1] * window
        dataSeg2[i,:] = y[int(start):int(end)+1] * window
        start+=D
        end+=D
    
    for i in range(K):
        fft1[i, :] = dofo(dataSeg1[i, :])
        fft2[i, :] = dofo(dataSeg2[i, :])
    U = 1/(np.sum(window * window)*fs)

    Ik = np.zeros((K, fft1.shape[1]//2+1), dtype=complex)
    for i in range(K):
        Ik[i, :] = fft2[i, :(fft2.shape[1]//2+1)] * np.conj(fft1[i, :(fft1.shape[1]//2+1)])

    csd = np.zeros((fft1.shape[1]//2+1), dtype=complex)
    for i in range(K):
        csd += Ik[i, :]
    csd= csd*U/K
    # As result is onesided, multiplying by 2 excluding DC (0) & Nyquist (L/2)
    csd[1:csd.shape[0]-1] = 2*csd[1:csd.shape[0]-1]

    return fn, csd

    