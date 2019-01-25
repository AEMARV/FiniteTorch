import numpy as np
import torch
import torch.functional
import torch.nn.functional as F
from torch.autograd import Function,Variable
import torch.nn
from typing import List,Tuple,Dict
from torch.distributions import Multinomial
from torch.distributions import Dirichlet
from torch import Tensor
#import klconvs
#import stochgrads
#import jconv
#import mdconv
#import argparse
import math
import random
import definition


class LogSumExp(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis,*args):
		input = input.detach()
		maxval = input.max(dim=axis, keepdim=True)[0]
		logsumexp = (input - maxval).exp().sum(dim=axis, keepdim=True).log() + maxval
		ctx.save_for_backward(input, logsumexp, Variable(input.new_tensor(axis, dtype=torch.int32)))
		return logsumexp
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		grad_output = grad_output.detach()
		input,lognorm,axis = ctx.saved_tensors
		prob = (input - lognorm).exp()
		grad_input = grad_output * prob
		return grad_input,None,None


def sample(lp:Tensor,axis=1,numsamples=1,MAP=False):
	lastaxis = lp.ndimension() -1
	lpt = lp.transpose(lastaxis,axis)
	M = Multinomial(total_count=numsamples,logits=lpt)
	#D = Dirichlet((lp.exp())*(numsamples.float())/(lp.size(lastaxis)))
	samps = M.sample().detach()
	samps = samps.transpose(lastaxis,axis)/numsamples
	logprob = (lp-(samps.detach()).log())
	logprob[logprob!=logprob] = float('Inf')
	logprob = logprob.min(dim=axis,keepdim=True)[0]

	logprob = logprob.sum(dim=(1,2,3),keepdim=True)
	return samps.detach(),logprob


def sample_mine(lp: Tensor, axis=1,numsamples=1):
	p = lp.exp()
	cumprob = p.cumsum(axis)
	sh = p.size()
	if axis == 1:
		rand = p.new_empty(sh[0], 1, sh[2], sh[3]).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
	else:
		rand = p.new_empty(sh[0], sh[1], sh[2], sh[3], 1).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0, high=1.0, size=(sh[0], sh[1], sh[2], sh[3],1))
	samps = cumprob > rand
	if axis == 1:
		samps[0:, 1:, 0:, 0:] = samps[0:, 1:, 0:, 0:] ^ samps[0:, 0:-1, 0:, 0:]
	elif axis == 4:
		samps[0:, 0:, 0:, 0:, 1:] = samps[0:, 0:, 0:, 0:, 1:] ^ samps[0:, 0:, 0:, 0:, 0:-1]
	else:
		raise (Exception(
			'Only axis=1 and axis=4(for binary distributions) is acceptable ' + 'axis=%d' % axis + ' was given'))
	samps = samps.type_as(p)
	return samps.detach(),None
def sampleprob_mine(p: Tensor, axis=1,numsamples=1):
	cumprob = p.cumsum(axis)
	sh = p.size()
	if axis == 1:
		rand = p.new_empty(sh[0], 1, sh[2], sh[3]).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
	else:
		rand = p.new_empty(sh[0], sh[1], sh[2], sh[3], 1).uniform_(0, 1)
	# np_rand = np.random.uniform(low=0.0, high=1.0, size=(sh[0], sh[1], sh[2], sh[3],1))
	samps = cumprob > rand
	if axis == 1:
		samps[0:, 1:, 0:, 0:] = samps[0:, 1:, 0:, 0:] ^ samps[0:, 0:-1, 0:, 0:]
	elif axis == 4:
		samps[0:, 0:, 0:, 0:, 1:] = samps[0:, 0:, 0:, 0:, 1:] ^ samps[0:, 0:, 0:, 0:, 0:-1]
	else:
		raise (Exception(
			'Only axis=1 and axis=4(for binary distributions) is acceptable ' + 'axis=%d' % axis + ' was given'))
	samps = samps.type_as(p)
	return samps.detach(),None

def sampleprob(p:Tensor,axis:int,numsamples):
	lastaxis = p.ndimension() - 1
	p = p.transpose(lastaxis, axis)
	M = Multinomial(total_count=numsamples, probs=p)
	# D = Dirichlet((lp.exp())*(numsamples.float())/(lp.size(lastaxis)))
	samps = M.sample()
	samps = samps.transpose(lastaxis, axis) / numsamples
	return samps,None


def sample_indpt(p:Tensor):
	sample = (p> torch.rand_like(p)).float()
	coef = 1/(p+1e-7)
	return sample


def sample_exp(p:Tensor):
	statenum = float(p.shape[1])
	bitnum = -(torch.rand_like(p[0:,0:1,0:1,0:1]).log()/math.log(2)).floor()

	torch.rand_like(p).log()/math.log(float(p.shape[1]))

	return None


def sample_maxprob(p,dum1,dum2):
	statenum = float(p.shape[1])
	print(p.sum(dim=1).mean().item())
	sample = (((p/p.sum(dim=1,keepdim=True))*statenum) >=1).float()
	#print(sample.sum(dim=1).mean().item())
	return sample,None
def sample_map(lp,axis,numsamps):
	maxlp,maxind = lp.max(dim=1,keepdim=True)
	boolind = (lp == maxlp).float()
	newlprob = (boolind/boolind.sum(dim=axis,keepdim=True)).log()
	samp,_ = sample(newlprob,axis=1,numsamples=numsamps)
	logprob = samp*lp
	logprob[logprob!=logprob] = 0
	return samp,logprob.sum(dim=axis,keepdim=True)


def p_map_backwardv1(lx:Tensor,lkernel:Tensor,lbias:Tensor,xsample:Tensor,log_ynorm,pad,stride,dzdynorm:Tensor):
	lkernel = lkernel.detach()
	lx = lx.detach()
	xsample = xsample.detach()
	lbias = lbias.detach()
	dzdy_forx_and_w = dzdynorm
	dzdy_forw_norm_component = (-log_ynorm.exp())*(dzdy_forx_and_w.sum(dim=1,keepdim=True))#*ynorm_sampled
	with torch.enable_grad():
		xsample.requires_grad = True
		ytemp = F.conv2d(xsample,lkernel*0 +1,padding=pad,stride=stride)
		dzdx, = torch.autograd.grad(ytemp,xsample,grad_outputs=dzdy_forx_and_w ,only_inputs=True)
	with torch.enable_grad():
		lkernel.requires_grad = True
		ytemp = F.conv2d(xsample,lkernel,padding=pad,stride=stride)
		dzdlw, = torch.autograd.grad(ytemp,lkernel,grad_outputs=dzdy_forx_and_w,only_inputs=True)
		dzdlw = dzdlw
	with torch.enable_grad():
		lkernel.requires_grad = True
		ytemp = F.conv2d(xsample,lkernel,padding=pad,stride=stride)
		dzdlw2, = torch.autograd.grad(ytemp,lkernel,grad_outputs=dzdy_forw_norm_component,only_inputs=True)
	dzdlx = -xsample * ((dzdx.abs().sum(dim=(1,2,3),keepdim=True))!=0).float()
	dzdlbias = ((dzdynorm -log_ynorm.exp()*(dzdynorm.sum(dim=1,keepdim=True)))).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
	dzdlbias = dzdlbias.sum(dim=0,keepdim=True)
	dzdlw = dzdlw + dzdlw2
	return dzdlx , dzdlw, dzdlbias


def p_map_backward_i(lx:Tensor,lkernel:Tensor,lbias:Tensor,kersample:Tensor,log_ynorm,trials,pad,stride,dzdynorm:Tensor):
	lkernel = lkernel.detach()
	lx = lx.detach()
	dzdy_forx_and_w = dzdynorm
	dzdy_forw_norm_component = (-log_ynorm.exp())*dzdy_forx_and_w.sum(dim=1,keepdim=True)#*ynorm_sampled
	with torch.enable_grad():
		lx.requires_grad = True
		ytemp = F.conv2d(lx,kersample,padding=pad,stride=stride)
		dzdx, = torch.autograd.grad(ytemp,lx,grad_outputs=dzdy_forx_and_w,only_inputs=True)

	with torch.enable_grad():
		lkernel.requires_grad = True
		ytemp = F.conv2d(lx*0 +1,lkernel,padding=pad,stride=stride)
		dzdlw, = torch.autograd.grad(ytemp,lkernel,grad_outputs=dzdy_forx_and_w,only_inputs=True)
	with torch.enable_grad():
		lkernel.requires_grad = True
		ytemp = F.conv2d(lx,kersample,padding=pad,stride=stride)
		dzdlx2, = torch.autograd.grad(ytemp,lkernel,grad_outputs=dzdy_forw_norm_component,only_inputs=True)
	dzdlx = dzdx + dzdlx2
	dzdlw = dzdlw *kersample

	return dzdlx , dzdlw


def p_map_backwardv2(lx:Tensor,lkernel:Tensor,lbias:Tensor,xsample:Tensor,log_ynorm,trials,pad,stride,dzdynorm:Tensor):
	dzdy_forx_and_w = dzdynorm
	dzdy_forw_norm_component = (-log_ynorm.exp())*dzdy_forx_and_w.sum(dim=1,keepdim=True)#*ynorm_sampled
	kernelsample,l = sample(lkernel,1,1)
	with torch.enable_grad():
		xsample.requires_grad = True
		kernelsample.requires_grad = True
		ytemp = F.conv2d(xsample,kernelsample,padding=pad,stride=stride)
		dzdx,dzdlw = torch.autograd.grad(ytemp,[xsample,kernelsample],grad_outputs=dzdy_forx_and_w,only_inputs=True)

	with torch.enable_grad():
		lkernel.requires_grad = True
		ytemp = F.conv2d(xsample,kernelsample,padding=pad,stride=stride)
		dzdlx2,dzdlw2 = torch.autograd.grad(ytemp,[xsample,kernelsample],grad_outputs=dzdy_forw_norm_component,only_inputs=True)
	dzdlx = (dzdx ) * lx.exp()
	dzdlbias = ((dzdynorm -log_ynorm.exp()*dzdynorm.sum(dim=1,keepdim=True))).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
	dzdlw = (dzdlw + dzdlw2*0) * lkernel.exp()

	return dzdlx , dzdlw, dzdlbias*0
