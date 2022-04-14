##########################################################################
#                            Tank Priyansu
#                            Python | 2022
#                      priyansutank3044@gmail.com
#
# This program is specifically the application of the Enhanced Frequency
# Domain Decomposition technique.
# It generates various plots extracting the modal properties of the system
# being under analysis.
#
# Reference:
# [1]  Brincker, R., Ventura, C. E.; Andersen, P. (2001). Damping estimation 
#      by frequency domain decomposition, Proceedings of the International 
#      Modal Analysis Conference - IMAC
##########################################################################

import numpy as np
import pandas as pd
from scipy import linalg as LA
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import mplcursors

def mac(m1, m2):
    MAC = np.abs(m1.conj().T @ m2)**2 / ((m1.conj().T @ m1)*(m2.conj().T @ m2))
    return MAC

def fdd(data, fs, resol=0.4, overlap=0.5, window='hann'):
    
    channels = data.shape[1]
    nPerSeg = fs/resol
    noverlap = nPerSeg * overlap
    maxFreq = fs//2

    gyy = np.zeros((channels, channels, int((nPerSeg/2)+1)), dtype=complex)

    for i in range(channels):
        for j in range(channels):
            f, pxy = signal.csd(data[:, i], data[:, j], window=window, fs=fs, nperseg=nPerSeg, noverlap=noverlap)
            gyy[i, j, :] = pxy
    # plt.plot(pxy)

    hxyt = np.zeros((channels, channels, int((nPerSeg/2)+1)), dtype=complex)
    gxx = np.zeros((channels, channels, int((nPerSeg/2)+1)))
    hxy = np.zeros((channels, channels, int((nPerSeg/2)+1)), dtype=complex)

    # fig, ax = plt.subplots()
    # for i in range (0, gyy.shape[0]):
    #     for j in range (0, gyy.shape[1]):
    #         # ax.plot(gyy[i, j,:])

    for i in range(gxx.shape[2]):
        u, s, v = LA.svd(gyy[:, :, i])
        gxx[:, :, i] = np.diag(s)
        hxyt[:, :, i] = u
        hxy[:, :, i] = u.T

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title('Singular Values Plot')
    figax = fig.add_subplot(111)
    for i in range(channels):
        figax.plot(f[:], np.log(gxx[i, i]), label='Singular Value {}'.format(i+1))
    figax.grid()
    figax.set_xlim(left=0, right=maxFreq)
    figax.xaxis.set_major_locator(MultipleLocator(maxFreq/10))
    figax.xaxis.set_minor_locator(MultipleLocator(maxFreq/100))
    figax.set_title("Singular values")
    figax.set_xlabel('Frequency [Hz]')
    figax.set_ylabel('dB $[g^2/Hz]$')
    figax.legend()

    mplcursors.cursor()

    Results={}
    Results['Data'] = {'Data' : data}
    Results['Data']['Samp. Freq.'] = fs
    Results['Data']['Freq. Resol.'] = resol
    Results['Singular Values'] = gxx
    Results['Singular Vectors'] = hxy
    Results['PSD Matrix'] = gyy

    return fig, Results

def FDDmode(FreQ, Results, ndf=2):
    fs = Results['Data']['Samp. Freq.']
    df = Results['Data']['Freq. Resol.']
    singVal = Results['Singular Values']
    singVec = Results['Singular Vectors']
    deltaf = ndf*df

    maxFreq = fs/2

    f = np.linspace(0, int(maxFreq), int(maxFreq*(1/df)+1))

    Freq = []
    index = []
    mode = []

    for freq in FreQ:
        lim = (freq - deltaf, freq + deltaf)
        idxLim = (np.argmin(abs(f-lim[0])), np.argmin(abs(f-lim[1])))

        s1overs2 = singVal[0, 0, idxLim[0]:idxLim[1]]/singVal[1, 1, idxLim[0]:idxLim[1]]
        maxs1s2 = np.max(s1overs2)
        idxmax = np.argmin(abs(s1overs2 - maxs1s2))
        idxmaxabs = idxLim[0] + idxmax
    #============================================================

        fidxmaxabs = f[idxmaxabs]
        modeShape = singVec[0, :, idxmaxabs]
        #normalize mode shape
        normModeShape = np.array(modeShape/modeShape[np.argmax(abs(modeShape))])

        Freq.append(fidxmaxabs)
        mode.append(normModeShape)
        index.append(idxmaxabs)

    Freq = np.array(Freq)
    mode = np.array(mode)
    index = np.array(index)

    Results = {}
    Results['Frequencies'] = Freq
    Results['Mode Shapes'] = mode.T
    Results['Freq. index'] = index

    return Results

def eFDDmode(FreQ, Results, ndf=2, MAC=0.85, skipPeaks=3, nPeaks=20, plot=False):
    data = Results['Data']['Data']
    fs = Results['Data']['Samp. Freq.']
    df = Results['Data']['Freq. Resol.']
    singVal = Results['Singular Values']
    singVect = Results['Singular Vectors']
    PSDmat = Results['PSD Matrix']

    fddmod = FDDmode(FreQ, Results, ndf=ndf)
    pickFreq, pickMode, pickIndex = fddmod['Frequencies'], fddmod['Mode Shapes'], fddmod['Freq. index']
    # print(fddmod)

    channels = data.shape[1]
    maxFreq = fs//2 #Nyquist freq
    spectLines = maxFreq/df + 1
    f = np.linspace(0, maxFreq, int(spectLines))
    invFFTpoints = int(spectLines)*20
    timelag = 1/df

    #Normalization of singVects
    singVectNorm = singVect.copy()
    for row in range(channels):
        for height in range(int(spectLines)):
            indx = np.argmax(abs(singVect[row, :, height]))
            singVectNorm[row, :, height] = singVectNorm[row, :, height]/singVect[row, indx, height]

    Freq = []
    modeShape = []
    damp = []

    #for each freq given
    for idx, freq in enumerate(pickFreq):
        pickModei = pickMode[:, idx]
        sdofSingVal = np.zeros(int(spectLines))
        sdofSingVec = np.zeros((channels, int(spectLines)), dtype=complex)
        # print(sdofSingVec.shape)

        #collect sval and svec maintaining MAC criteria
        #on increasing side
        x=0
        while mac(pickModei, singVectNorm[0, :, pickIndex[idx] + x]) > MAC:
            sdofSingVal[pickIndex[idx] + x] = singVal[0, 0, pickIndex[idx] + x]
            sdofSingVec[:, pickIndex[idx] + x] = singVectNorm[0, :, pickIndex[idx] + x]
            x+=1
        while mac(pickModei, singVectNorm[1, :, pickIndex[idx] + x]) > MAC:
            sdofSingVal[pickIndex[idx] + x] = singVal[1, 1, pickIndex[idx] + x]
            sdofSingVec[:, pickIndex[idx] + x] = singVectNorm[1, :, pickIndex[idx] + x]
            x+=1   
        # on decreasing side
        x=0
        while mac(pickModei, singVectNorm[0, :, pickIndex[idx] + x]) > MAC:
            sdofSingVal[pickIndex[idx] + x] = singVal[0, 0, pickIndex[idx] + x]
            sdofSingVec[:, pickIndex[idx] + x] = singVectNorm[0, :, pickIndex[idx] + x]
            x-=1
        while mac(pickModei, singVectNorm[1, :, pickIndex[idx] + x]) > MAC:
            sdofSingVal[pickIndex[idx] + x] = singVal[1, 1, pickIndex[idx] + x]
            sdofSingVec[:, pickIndex[idx] + x] = singVectNorm[1, :, pickIndex[idx] + x]
            x-=1

        #finding idx where sdofval is collected
        idxsdofbell = np.squeeze(np.array(np.where(sdofSingVal)))
        fidxsdofbell = f[idxsdofbell]

        #mode shape associated to each singval and weighted by singval itself
        sdofSingVecw = np.array([np.array(sdofSingVal[idxsdofbell[i]] * sdofSingVec[:, idxsdofbell[i]]) for i in range(len(idxsdofbell))])
        sdofSingVecw = np.squeeze(sdofSingVecw)

        #normalized mode shape
        meanSdofVecw = np.mean(sdofSingVecw, axis=0)
        normSdofVecw = meanSdofVecw/meanSdofVecw[np.argmax(np.abs(meanSdofVecw))]

        #Autocorrelation Function (Free Decay)
        sdofCor = np.fft.ifft(sdofSingVal, n=invFFTpoints, axis=0, norm='ortho').real
        tau = np.linspace(0, timelag, len(sdofCor))

        #normalize auto correlation
        normSdofCor = sdofCor[:len(sdofCor)//2] / sdofCor[np.argmax(sdofCor)] #only one side taken

        #roots finding
        sgn = np.sign(normSdofCor)
        sgnChange = np.diff(sgn)
        zeroCross = np.where(sgnChange)[0]

        #finding peaks (max and min) of auto correlation
        maxSdofCor = [np.max(normSdofCor[zeroCross[i]:zeroCross[i+2]]) for i in range(0, len(zeroCross)-2, 2)]
        minSdofCor = [np.min(normSdofCor[zeroCross[i]:zeroCross[i+2]]) for i in range(0, len(zeroCross)-2, 2)]
        if len(maxSdofCor) > len(minSdofCor):
            maxSdofCor = maxSdofCor[:-1]
        elif len(minSdofCor) > len(maxSdofCor):
            minSdofCor = minSdofCor[:-1]
        peaks = np.ravel(np.array((minSdofCor, maxSdofCor)), order='F')

        #finding idx of peaks
        maxSdofCorIdx = [np.argmin(abs(normSdofCor - x)) for x in maxSdofCor]
        minSdofCorIdx = [np.argmin(abs(normSdofCor - x)) for x in minSdofCor]
        peaksIdx = np.ravel(np.array((minSdofCorIdx, maxSdofCorIdx)), order='F')

        #peaks and idx of peaks used in curve fitting
        peaksFit = np.array(peaks[skipPeaks:skipPeaks+nPeaks])
        peaksFitidx = np.array(peaksIdx[skipPeaks:skipPeaks+nPeaks])

        #estimating natural freq
        fd = 1/(np.mean(np.diff(tau[peaksFitidx])*2))
        
        #log decrement
        deltak = np.array([2*np.log(np.abs(peaks[0])/np.abs(peaks[i])) for i in range(len(peaksFit))])

        #curve fitting and finding logarithmix decrement
        func = lambda x,delta : x*delta
        delta, o = curve_fit(func, np.arange(len(peaksFit)), deltak)

        #damping ratio and natural freq
        xi = delta/(np.sqrt(delta**2 + 4*np.pi**2))
        fn = fd/np.sqrt(1-xi**2)

        Freq.append(fn)
        damp.append(xi)
        modeShape.append(normSdofVecw)

        #plots
        if plot:
            fig = plt.figure(figsize=(16, 8))
            fig.canvas.manager.set_window_title('Mode {}'.format(idx+1))
            gs1 = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
            gs2 = gridspec.GridSpec(2, 3, height_ratios=[1, 2])
            ax1 = fig.add_subplot(gs1[0])
            ax2 = fig.add_subplot(gs2[3])
            ax3 = fig.add_subplot(gs2[4])
            ax4 = fig.add_subplot(gs2[5])

            #singular value
            ax1.plot(f, 10*np.log10(singVal[0, 0]), c='b', label='Singular Values')
            ax1.plot(fidxsdofbell, 10*np.log10(sdofSingVal[idxsdofbell].real), c='r', label='SDOF Bell')
            ax1.set_title("Estimated SDOF Bell")
            ax1.set_xlabel("Frequency [Hz]")
            ax1.set_ylabel('dB $[g^2/Hz]$')
            ax1.legend()
            ax1.grid()
            
            #sdof autocorrelation
            ax2.plot(tau[:len(sdofCor)//2], normSdofCor, c='b')
            ax2.set_title('SDOF Auto correlation Function')
            ax2.set_xlabel(r'$\tau$ [s]')
            ax2.set_ylabel('Normalized Correlation')
            ax2.grid()

            #sdof autocorrelation for fit
            ax3.plot(tau[:peaksFitidx[-1]], normSdofCor[:peaksFitidx[-1]], c='b', zorder=10)
            ax3.scatter(tau[peaksFitidx], normSdofCor[peaksFitidx], c='r', zorder=20)
            ax3.set_title('Auto correlation Function used for Curve Fitting')
            ax3.set_xlabel(r'$\tau$ [s]')
            ax3.set_ylabel('Normalized Correlation')
            ax3.grid()

            #fitted damping
            ax4.plot(np.arange(len(peaksFit)), delta*np.arange(len(peaksFit)), c='b', zorder=10)
            ax4.scatter(np.arange(len(peaksFit)), deltak, c='r', zorder=20, marker="x")
            ax4.set_title('Fitted Damping by Logarithmic Decrement\n' + r'$f_n$ = {0:1.2f}, $f_d$ = {1:1.2f}, $\xi$ = {2:1.2f}%'.format(float(fn), fd, float(xi*100)))
            ax4.set_xlabel(r'$k^{th}$ Peak')
            ax4.set_ylabel(r'$2ln\left(\frac{r_0}{|r_k|}\right)$')
            ax4.grid()

            plt.tight_layout()

    Freq = np.array(Freq)
    damp = np.array(damp)
    modeShape = np.array(modeShape)

    Results = {}
    Results['Frequencies'] = Freq
    Results['Damping'] = damp
    Results['Mode Shapes'] = modeShape.T

    return Results

def filterData(data, fs, q, high):
    data = signal.detrend(data, axis=0)
    data = signal.decimate(data, q, ftype='fir', axis=0)
    fs = fs/q
    for i in range(data.shape[1]):
        b, a = butter(15, high, fs=fs, btype='lowpass')
        data[:, i] = lfilter(b, a, data[:, i])
    return data, fs

#============================================================
#data
data = pd.read_csv("d:/PRIYANSU/PRL/data_loop/fine_data/pier_hammering_5m.csv", header=0, sep=",")
data = data.to_numpy()
data = data[:, 1:]

fs = 1612

data, fs = filterData(data, fs, 2, 380)

#============================================================

FDD = fdd(data, fs)
FreQ = [36.4, 39.16, 97.2, 151.35]
# FreQ = [2.03, 5.91, 9.34, 11.99, 13.69]
resEFDD = eFDDmode(FreQ, FDD[1], plot=True)

plt.show()
