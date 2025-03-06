from MFDFA import singspect
from MFDFA import  MFDFA
import numpy as np
#import numba
#from numba import jit, prange
from scipy.optimize import curve_fit
import pandas as pd
from scipy.special import expit

import warnings
warnings.filterwarnings("ignore")


def getDeltaAlpha(alpha,falpha):
	a,b,c = getPolynomial(alpha,falpha)
	if a == 0:
		return 0
	if b*b<4*a*c:
		return 0
	# Bahskara to find the roots of the fitted polynomial
	return np.power(np.sqrt(b*b-4*a*c)/a,2) 
	
#def deltaAlpha(alpha):
#	return np.average(np.max(alpha,axis=1)-np.min(alpha,axis=1))

def boxSymmetry(alpha,falpha):
	mx,mn = np.argmax(alpha),np.argmin(alpha)
	return np.abs((np.max(alpha)-np.min(alpha))*(falpha[mx]-falpha[mn]))
	
def polyNormalization(x):
	return x/(1+x)

def singularitySpectrumMetrics(alpha,falpha):
	'''
	========================================================================
	Measures of the singularity spectrum
	========================================================================
	Input:
	alpha - x values of the singularity spectrum (np.array that must have same lenght of falpha)
	falpha - y values of the singularity spectrum (np.array that must have same lenght of alpha)
	========================================================================
	Output:
	Dictionay with the measures delta_alpha,max_f, delta_f and asymmetry
	========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	maxFa = np.argmax(falpha)
	delta = np.max(alpha)-np.min(alpha)
	assym = np.inf if np.abs(falpha[0]-falpha[len(falpha)-1])<1e-15 else np.abs(falpha[0]-falpha[len(falpha)-1])
    
	return {'delta_alpha':delta,
		'max_f':falpha[maxFa],
		'delta_f': (np.max(falpha)-np.min(falpha)),
		'asymmetry': assym,
		'alpha':alpha,
		'falpha':falpha
		}

def getAverageSing(serie):
	'''
	========================================================================
	Retrieves the singularity spectrum with median delta alpha
	========================================================================
	Input:
	serie - time series
	========================================================================
	Output:
	alphas and f(alphas)
	========================================================================
	Wrote by: Rubens A. Sautter (12/2022)
	'''
	a, fa, lda = autoMFDFA(serie,nqs=20)		
	metrics = [singularitySpectrumMetrics(a[i],fa[i]) for i in range(len(a))]
	return np.average(a,axis=0), np.average(fa,axis=0)

def selectScales(timeSeries,threshold=1e-3,nscales=30):
	'''
	========================================================================
	Select random scales to apply MFDFA, from a set of frequencies with  
	large Power Spectrum Density values 
	========================================================================
	Input:
	timeSeres - input time series (np.array)
	threshold - determines the minimum PSD of the series (0 to 1)
	========================================================================
	Output:
	scales - set of scales randomly selected
	========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	psd = np.fft.fft(timeSeries)
	freq = np.fft.fftfreq(len(timeSeries))
	psd = np.real(psd*np.conj(psd))
	pos = (freq>1e-13)
	psd = psd[pos]
	freq = freq[pos]
	maxPSD = np.max(psd)
	psd = psd/maxPSD
	scales = 1/np.abs(freq[(psd >threshold)])
	scales = scales.astype(int)
	scales = np.unique(scales)
	scales = np.sort(scales)
	return np.random.choice(scales,nscales)

def normalize(d):
	data = d-np.average(d)
	data = data/np.std(data)
	return data

#@jit(forceobj=True,parallel=True)
def autoMFDFA(timeSeries,qs=np.arange(5,15,2), scThresh=1e-2, nqs = 10, nsamples=40, nscales=20):
	'''
	========================================================================
	Complementary method to measure multifractal spectrum.
	Base MFDFA implementation: https://github.com/LRydin/MFDFA

	(I)	The time series is normalized according to its global average and global standard deviation
	(II)	A set of scales is randomly selected	
	(III)	MFDFA is applied over the normalized time-series  ('nsamples' times) 
	(IV)	Delta Alpha is measured
	(V)	Singularity spectrum with delta alpha outliers (greater than average + standard deviation) are removed 
	(VI)	logistic function is measured
	=========================================================================
	Input:
	timeSeries - serie of elements (np.array)
	qs - set of hurst exponent ranges
	scThresh - threshold to select DFA scales (see selectScales function)
	nqs - number of hurst exponents measured
	nsamples - number of singularity spectrum samples per hurst exponent set (q)
	nscales - number of random scales
	=========================================================================
	Output:
	alphas, falphas - set of multifractal spectrum
	Average delta alpha
	=========================================================================
	Wrote by: Rubens A. Sautter (02/2022)
	'''
	
	# signularity spectra of the series
	alphas,falphas = [], []
	
	# signularity spectra of surrogate series
	salphas,sfalphas = [], []
	
	data = normalize(timeSeries)
	
	deltas = []
	 
	for i in range(nsamples):
		scales = selectScales(data,threshold=scThresh,nscales=nscales)
		for it  in range(len(qs)):
			qrange = qs[it]
			q = np.linspace(-qrange,qrange,nqs)
			q = q[q != 0.0]
			lag,dfa = MFDFA(data, scales, q=q)
			alpha,falpha = singspect.singularity_spectrum(lag,dfa,q=q)
			if np.isnan(alpha).any() or np.isnan(falpha).any():
				continue
			if (falpha>1.5).any():
				continue
			alphas.append(alpha)
			falphas.append(falpha)
			deltas.append(boxSymmetry(alpha,falpha))
			
	
	seq = np.argsort(deltas)
	# Remove outlier:
	alphas,falphas = np.array(alphas),np.array(falphas) 
	alphas,falphas = alphas[seq],falphas[seq]
	alphas,falphas = alphas[:len(alphas)//2],falphas[:len(alphas)//2]

	#compute ldas:
	ldas = []
	for f in range(len(alphas)):
		ldas.append(polyNormalization(getDeltaAlpha(alphas[f],falphas[f])))

	if len(alphas)>2:
		return alphas, falphas, np.average(ldas)
	else:
		return alphas, falphas, 0

