##########################################################################
#                            Tank Priyansu
#                            Python | 2022
#                      priyansutank3044@gmail.com
#
# Func fdd implements all the developed codes from scratch for doing FDD.
# Func fddD implements ready to use and optimized codes for FFT from scipy.
##########################################################################

from numpy import linalg as la
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal

import numpy as np

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

    L = 512

    fn = np.arange(L*32//2+1)*fs/(L*32)

    window = hannWindow(L)

    D = 256
    K = int(1 + (len(x) - L)//D)

    start = 0
    end = start + L - 1

    dataSeg1 = np.zeros((K, L), dtype=complex)
    dataSeg2 = np.zeros((K, L), dtype=complex)
    fft1 = np.zeros((K, L*32), dtype=complex)
    fft2 = np.zeros((K, L*32), dtype=complex)
    print('r')
    for i in range(K):
        dataSeg1[i,:] = x[int(start):int(end)+1] * window
        dataSeg2[i,:] = y[int(start):int(end)+1] * window
        start+=D
        end+=D
    
    for i in range(K):
        fft1[i, :] = np.hstack((dofo(dataSeg1[i, :]), np.zeros((512*31), dtype=complex)))
        fft2[i, :] = np.hstack((dofo(dataSeg2[i, :]), np.zeros((512*31), dtype=complex)))
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

def fdd(data, fs):
    channel = data.shape[1]
    nqFreq = fs/2

    psdMat = np.zeros((channel, channel, 512*32//2+1), dtype=complex)
    singVals = np.zeros((channel, channel, 512*32//2+1))
    singVecs = np.zeros((channel, channel, 512*32//2+1), dtype=complex)

    for i in range(channel):
        for j in range(channel):
            f, Pxy = crossSpectral(data[:, i], data[:, j], fs)
            psdMat[i, j, :] = Pxy
        plt.plot(f, psdMat[i, j, :])
        # plt.plot(fn, psdMatn[i, j, :])
    for i in range(psdMat.shape[2]):
        U, S, Vt = np.linalg.svd(psdMat[:, :, i])
        singVals[:, :, i] = np.diag(S)
        singVecs[:, :, i] = U.T
    
def fddD(data, fs):
    ##########################################################################
    # Input Parameters:
    # data: Accelerometer reading data as np array of size (samples X channels)
    # fs: sampling rate of data
    # 
    # Output Parameters:
    # f: Sampling frequencies
    # gxx: PSD matrix of input
    # hxy: Frequency Response Function (FRF) matrix
    # gyy: PSD matrix of output
    ##########################################################################

    channels = data.shape[1]
    gyy = np.zeros((channels, channels, 5000+1), dtype=complex)

    for i in range(channels):
        for j in range(channels):
            f, pxy = signal.csd(data[:, i], data[:, j], fs=fs, nperseg=10000, noverlap=5000)
            gyy[i, j, :] = pxy
    
    hxyt = np.zeros((channels, channels, 5000+1), dtype=complex)
    gxx = np.zeros((channels, channels, 5000+1))
    hxy = np.zeros((channels, channels, 5000+1), dtype=complex)

    for i in range(gxx.shape[2]):
        u, s, v = la.svd(gyy[:, :, i])
        gxx[:, :, i] = np.diag(s)
        hxyt[:, :, i] = u
        hxy[:, :, i] = v.T 
    
    return f, gxx, hxy, gyy


#####################################################################
# Example data 
data = pd.read_csv("data.csv", header=0, sep=",", index_col=False)
data = data.to_numpy()

fs = 500

data = signal.detrend(data, axis=0)
f, gxx, hxy, gyy = fddD(data, fs)

#plot
fig, ax = plt.subplots()
for i in range(data.shape[1]):
    ax.plot(f, 10*np.log10(gxx[i, i, :])) #db
ax.grid()
ax.set_xlim(left = 0, right = fs//2)
plt.show()