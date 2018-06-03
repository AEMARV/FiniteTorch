import numpy as np
import torch
import torch.functional
from torch.autograd import Function,Variable
import torch.nn
from typing import List,Tuple,Dict
from definition import epsilon
import torch.nn.functional as F
from torch import Tensor
import argparse

class LogSumExpStoch(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis):
		input = input.detach()
		maxval = input.max(dim=axis,keepdim=True)[0]
		logsumexp = (input - maxval).exp().sum(dim=axis,keepdim=True).log() + maxval
		test= input - logsumexp
		testprob = test.exp()
		testprob = testprob.sum(dim=axis,keepdim=False)

		ctx.save_for_backward(input,logsumexp,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return logsumexp
	@staticmethod
	def backward(ctx, grad_output:List[torch.Tensor]):
		grad_output = grad_output.detach()
		input,lognorm,axis = ctx.saved_tensors
		prob = (input - lognorm).exp()
		samps = LogSumExpStoch.sample(prob,axis.item())
		grad_input = grad_output * samps
		return grad_input,None

	@staticmethod
	def sample(p:Tensor,axis):
		cumprob = p.cumsum(axis)
		sh = p.size()
		if axis==1 :
			rand = p.new_empty(sh[0],1,sh[2],sh[3]).uniform_(0,1)
			#np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
		else:
			rand = p.new_empty(sh[0], sh[1], sh[2], sh[3],1).uniform_(0,1)
			#np_rand = np.random.uniform(low=0.0, high=1.0, size=(sh[0], sh[1], sh[2], sh[3],1))
		samps = cumprob > rand
		if axis == 1:
			samps[0:, 1:, 0:, 0:] = samps[0:, 1:, 0:, 0:] ^ samps[0:,0:-1,0:,0:]
		elif axis == 4:
			samps[0:, 0:, 0:, 0:, 1:] = samps[0:, 0:, 0:, 0:, 1:] ^ samps[0:,0:,0:,0:,0:-1]
		else:
			raise(Exception('Only axis=1 and axis=4(for binary distributions) is acceptable '+ 'axis=%d'%axis+ ' was given'))
		samps = samps.type_as(p)
		return samps
class Sampler(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis):
		input = input.detach()
		inputp = input.exp()
		samples = Sampler.sample(inputp,axis)
		ctx.save_for_backward(input,samples,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return inputp
	@staticmethod
	def backward(ctx, grad_output:List[torch.Tensor]):
		#grad_output = grad_output.detach()
		input, samples, axis = ctx.saved_tensors
		inputp = input.exp()
		samplesnew = Sampler.sample(inputp, axis)
		return grad_output* samplesnew,None

	@staticmethod
	def sample(p:Tensor,axis):
		cumprob = p.cumsum(axis)
		sh = p.size()
		if axis==1 :
			rand = p.new_empty(sh[0],1,sh[2],sh[3]).uniform_(0,1)
			#np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
		else:
			rand = p.new_empty(sh[0], sh[1], sh[2], sh[3],1).uniform_(0,1)
			#np_rand = np.random.uniform(low=0.0, high=1.0, size=(sh[0], sh[1], sh[2], sh[3],1))
		samps = cumprob > rand
		if axis == 1:
			samps[0:, 1:, 0:, 0:] = samps[0:, 1:, 0:, 0:] ^ samps[0:,0:-1,0:,0:]
		elif axis == 4:
			samps[0:, 0:, 0:, 0:, 1:] = samps[0:, 0:, 0:, 0:, 1:] ^ samps[0:,0:,0:,0:,0:-1]
		else:
			raise(Exception('Only axis=1 and axis=4(for binary distributions) is acceptable '+ 'axis=%d'%axis+ ' was given'))
		samps = samps.type_as(p)
		return samps
class LogSumExp(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis):
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
		return grad_input,None





