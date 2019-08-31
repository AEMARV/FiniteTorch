import numpy as np
import scipy as sp
import typing
import math
def sample(p:np.ndarray,numsamp=1):
	cumprob = p.cumsum(axis=0)
	rands = np.random.rand(1,numsamp)
	samps = rands<cumprob
	samps[1:,0:] = samps[0:-1,0:] ^ samps[1:,0:]
	lp = np.log((samps * p).sum(axis=0, keepdims=True))
	return samps , lp

def annealed_sampling(p,alpha,numsamp=1):
	''' Extends rejection sampling by changing the range of uniform random values generated
	FAIL.
	'''
	running_log =0
	accepted = np.zeros((1,numsamp),dtype=bool)
	outputsamp = np.zeros((p.shape[0],numsamp))
	while(not (accepted).all()):
		samp =sample(p,numsamp=numsamp) # type:np.ndarray
		lprobsamp= np.log((samp*p).sum(axis=0,keepdims=True))*(alpha-1)
		running_log= np.log(np.random.rand(1,samp.shape[1]))+running_log
		new_accepted = (running_log <= lprobsamp) & (~accepted)
		accepted = new_accepted | accepted
		outputsamp[0:,new_accepted.squeeze()] = samp[0:,new_accepted.squeeze()]
	return outputsamp

def reject_sampling(p,alpha,numsamp=1):
	''' Extends rejection sampling by changing the range of uniform random values generated
	FAIL.


	'''

	def randgen(): return np.log(np.random.rand(1, numsamp))
	running_log =0
	accepted = np.zeros((1,numsamp),dtype=bool)
	outputsamp = np.zeros((p.shape[0],numsamp))
	while(not (accepted).all()):
		samp,lprobsamp =sample(p,numsamp=numsamp) # type:np.ndarray
		new_accepted = ((randgen()*2) <= lprobsamp) & (~accepted)
		accepted = new_accepted | accepted
		outputsamp[0:,new_accepted.squeeze()] = samp[0:,new_accepted.squeeze()]
	return outputsamp

def slice_sampling(p,alpha,numsamp=1):
	''' The range of the uniform drops to the probability of the examined state
	Fail
	'''

	def randgen(): return np.log(np.random.rand(1, numsamp))
	def swap(a,b,logic):
		temp = a
		a[logic] = b[logic]
		b[logic] = temp[logic]
		return a,b
	def swap_samp(a,b,logic):
		temp = a
		a[0:,logic] = b[0:,logic]
		b[0:,logic] = temp[0:,logic]
		return a, b
	accepted = np.zeros((1,numsamp),dtype=bool)
	outputsamp = np.zeros((p.shape[0],numsamp))
	iter= 0
	anchorsamp, anchorprob = sample(p, numsamp=numsamp)  # type:np.ndarray
	anchorprob *= alpha-1
	temp = anchorprob
	while(not (accepted).all()):
		samp, lower =sample(p,numsamp=numsamp) # type:np.ndarray
		samp2, upper= sample(p,numsamp=numsamp) # type:np.ndarray
		lower,upper= swap(lower,upper,upper<lower)
		zeroband = randgen() < (lower-upper)

		upper[zeroband] = lower[zeroband]
		lower[zeroband] = -np.inf

		samp,lprobsamp = sample(p,numsamp=numsamp) # type:np.ndarray

		probsamp = np.exp(lprobsamp)
		acceptprob = (probsamp - np.exp(lower))/(np.exp(upper) - np.exp(lower))
		new_accepted = (np.exp(randgen()) < acceptprob )& (~accepted)
		accepted = new_accepted | accepted
		outputsamp[0:,new_accepted.squeeze()] = samp[0:,new_accepted.squeeze()]
		iter +=1
	return outputsamp



if __name__ == '__main__':
	alpha=2
	numsamp = 100000
	p = np.random.rand(100,1)
	p = p
	p = p / p.sum(axis=0,keepdims=True)
	q = p**alpha
	q = q/ q.sum(axis=0,keepdims=True)
	samps = slice_sampling(p,alpha,numsamp=numsamp)

	alpha_emp_prob = samps.mean(axis=1)
	print(np.abs((q.squeeze()-alpha_emp_prob.squeeze())).sum())


	samps,_ = sample(q,numsamp)
	emp_prob = samps.mean(axis=1)
	print(np.abs((q.squeeze() - emp_prob.squeeze())).sum())
