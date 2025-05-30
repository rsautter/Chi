from .spectralEntropy import *
from .singularitySpectrum import *
from .qqMetric import *
from .datasetFinder import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns 
from matplotlib.lines import Line2D

def chiSpace(data,qs=np.arange(5,15,2), scThresh=1e-2, nqs = 10, nsamples=40, nscales=20):
	'''
	========================================================================
	
	Generates an dictionary with the Gaussian Distance metric (GQQ) and the Logistic Delta Alpha (LDA)
	
	========================================================================
	Input:
	data - time series (0D+1)
	qs - Hurst exponents extremes
	scThresh - scale threshold of autoMFDFA
	nqs - number of hurst exponents
	nsamples - number of scale samples to compute delta alpha
	========================================================================
	Output:
	Dictionary with keywords 'GQQ' and 'Alpha', where:
	GQQ - Gaussian Q-Q Plot distance
	Alpha - Fitted Singularity Spectra Delta Alpha Normalized by: x/(1+x) 
	========================================================================
	Wrote by: Rubens A. Sautter (08/2022)
	'''
	_,_, alpha = autoMFDFA(data,qs=qs,scThresh=scThresh,nqs=nqs,nsamples=nsamples, nscales=nscales)
	return {"GQQ":qqGaussianDistance(data), "Alpha":alpha}
	
def bootstrapChi(data,bsamples=np.linspace(0.8,1.0,20),qs=np.arange(5,15,2), scThresh=1e-2, nqs = 10, nsamples=40, nscales=20):
	'''
	========================================================================
	
	Bootstraps the time series around the peak to find the distribution of Gaussian Distance metric (GQQ) and the Logistic Delta Alpha (LDA)
	returns a pandas dataframe
	
	========================================================================
	Input:
	data - time series (0D+1)
	bsamples  - percentage to bootstrap the series around the maximum value (must be a list of floatoing points in the interval ]0,1] )
	qs - Hurst exponents extremes
	scThresh - scale threshold of autoMFDFA
	nqs - number of hurst exponents
	nsamples - number of scale samples to compute delta alpha
	========================================================================
	Output:
	Dictionary with keywords 'GQQ' and 'alpha', where:
	GQQ - Gaussian Q-Q Plot distance
	Alpha - Fitted Singularity Spectra Delta Alpha Normalized by: x/(1+x) 
	========================================================================
	Wrote by: Rubens A. Sautter (04/2023)
	'''
	peak = np.argmax(np.abs(data))
	samples = []
	for p in bsamples:
		npoints = int(p*len(data))
		if peak>len(data)//2:
			maxPoint = min(peak+npoints//2,len(data))
			minPoint = maxPoint-npoints
		else:
			minPoint = max(0,peak-npoints//2)
			maxPoint = peak + npoints -minPoint
		samples.append(chiSpace(data[minPoint:maxPoint],qs,scThresh,nqs,nsamples,nscales))
	return pd.DataFrame(samples)
	
def segmentPeak(data,windowSize=0.25):
	'''
	========================================================================
	
	Selects the region around the largest peak of the time series.
	The size of the region is determined as a percentage of the original series
	
	========================================================================
	Input:
	data - time series (0D+1)
	windowSize  - percentage of the time series select around the peak 
	========================================================================
	Output:
	Series centralized
	========================================================================
	Wrote by: Rubens A. Sautter (04/2023)
	'''
	peak = np.argmax(np.abs(data))
	samples = []
	npoints = int(windowSize*len(data))
	if peak>len(data)//2:
		maxPoint = min(peak+npoints//2,len(data))
		minPoint = maxPoint-npoints
	else:
		minPoint = max(0,peak-npoints//2)
		maxPoint = peak + npoints -minPoint
	return data[minPoint:maxPoint]
		
def plot(figsize=(12,12)):
	'''
	========================================================================
	Plots the Chi-Space using matplotlib and seaborn, it also shows the histogram of GQQ and Delta Alpha
	========================================================================
	figsize - size of of the figure frame
	========================================================================
	Wrote by: Rubens A. Sautter (12/2022)
	'''
	# Loading data	
	endo = get_endoDataResults()
	exo = get_exoDataResults()
	reds = get_redsDataResults()
	zL = get_LorenzDataResults()
	
	# Plotting
	
	gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1],wspace=0,hspace=0)

	plt.figure(figsize=figsize)

	plt.subplot(gs[0,0])
	sns.kdeplot(data=zL, x="Alpha", y="GQQ",fill=True,color='goldenrod',thresh=0.02,alpha = 0.3)
	sns.kdeplot(data=endo, x="Alpha", y="GQQ",fill=True,color='cyan',thresh=0.05,alpha = 0.5)
	sns.kdeplot(data=exo, x="Alpha", y="GQQ",fill=True,color='blue',thresh=0.005,alpha = 0.7)
	sns.kdeplot(data=reds, x="Alpha", y="GQQ",fill=True,color='red',thresh=0.05,alpha = 0.5)

	#wn = plt.scatter(0.0,0.0,s=200,color='k')

	plt.xlim(0.0,1.0)
	plt.ylim(0.0,1.0)
	handles = [Line2D([0], [0], color='cyan', lw=6),
		             Line2D([0], [0], color='blue', lw=6),
		             Line2D([0], [0], color='red', lw=6),
		             Line2D([0], [0], color='goldenrod', lw=6)
		             ]
	labels = ['Endogenous p-model','Exogenous p-model','Red Noise','Lorenz Synchronization']
	plt.legend(handles,labels,loc=2)
	plt.xlabel('')
	plt.xticks([])
	plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	plt.ylabel(r"$G_{QQ}$",fontsize=20)


	plt.subplot(gs[0,1])
	ax = sns.histplot(reds,color='red',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(zL,color='goldenrod',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(exo,color='blue',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(endo,color='cyan',fill=False,y='GQQ',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()

	plt.ylim(0,1)
	plt.yticks([])
	plt.ylabel('')
	plt.xticks([5,10,15,20,25],["5%","10%","15%","20%","25%"])
	plt.xlabel('')


	plt.subplot(gs[1,0])
	ax = sns.histplot(reds,color='red',fill=False,x='Alpha',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(zL,color='goldenrod',fill=False,x='Alpha',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(exo,color='blue',fill=False,x='Alpha',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()
	ax = sns.histplot(endo,color='cyan',fill=False,x='Alpha',lw=3.5,bins=10,stat="percent",kde=True,kde_kws={'cut': 3})
	ax.containers[0].remove()

	plt.xlim(0,1)
	plt.yticks([10,20,30,40,50],["10%","20%","30%","40%","50%"])
	plt.xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	plt.ylabel('')
	plt.xlabel(r"$$\Delta \alpha_N$",fontsize=20)
	plt.tight_layout()
	
def plot2(figsize=(12,12)):
	'''
	========================================================================
	Plots the Chi-Space using matplotlib and seaborn
	========================================================================
	figsize - size of of the figure frame
	========================================================================
	Wrote by: Rubens A. Sautter (12/2022)
	'''
	# Loading data	
	endo = get_endoDataResults()
	exo = get_exoDataResults()
	reds = get_redsDataResults()
	zL = get_LorenzDataResults()
	
	# Plotting
	
	gs = GridSpec(1, 1,wspace=0,hspace=0)

	plt.figure(figsize=figsize)

	plt.subplot(gs[0,0])
	sns.kdeplot(data=zL, x="Alpha", y="GQQ",fill=True,color='goldenrod',thresh=0.02,alpha = 0.3)
	sns.kdeplot(data=endo, x="Alpha", y="GQQ",fill=True,color='cyan',thresh=0.05,alpha = 0.5)
	sns.kdeplot(data=exo, x="Alpha", y="GQQ",fill=True,color='blue',thresh=0.005,alpha = 0.7)
	sns.kdeplot(data=reds, x="Alpha", y="GQQ",fill=True,color='red',thresh=0.05,alpha = 0.5)

	#wn = plt.scatter(0.0,0.0,s=200,color='k')

	plt.xlim(0.0,1.0)
	plt.ylim(0.0,1.0)
	handles = [Line2D([0], [0], color='cyan', lw=6),
		             Line2D([0], [0], color='blue', lw=6),
		             Line2D([0], [0], color='red', lw=6),
		             Line2D([0], [0], color='goldenrod', lw=6)
		             ]
	labels = ['Endogenous p-model','Exogenous p-model','Red Noise','Lorenz Synchronization']
	plt.legend(handles,labels,loc=2)
	plt.xlabel('')
	plt.xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
	plt.ylabel(r"$G_{QQ}$",fontsize=20)
	plt.xlabel(r"$\Delta \alpha_N$",fontsize=20)
	
	return gs, [handles,labels]
