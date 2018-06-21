import numpy as np
import torch
import torch.functional
from torch.autograd import Function,Variable
import torch.nn
from typing import List,Tuple,Dict
from torch.distributions import Multinomial
from torch.distributions import Dirichlet
from torch import Tensor
import argparse
def sample(lp:Tensor,axis:int,numsamples):
	lastaxis = lp.ndimension() -1
	lp = lp.transpose(lastaxis,axis)
	M = Multinomial(total_count=numsamples,logits=lp)
	#D = Dirichlet((lp.exp())*(numsamples.float())/(lp.size(lastaxis)))
	samps = M.sample()
	samps = samps.transpose(lastaxis,axis)/numsamples
	return samps
class LogSumExpStoch(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis,totalcount):
		input = input.detach()
		maxval = input.max(dim=axis,keepdim=True)[0]
		logsumexp = (input - maxval).exp().sum(dim=axis,keepdim=True).log() + maxval
		ctx.save_for_backward(input,logsumexp,Variable(input.new_tensor(axis,dtype=torch.int32)),Variable(input.new_tensor(totalcount,dtype=torch.int32)))
		return logsumexp
	@staticmethod
	def backward(ctx, grad_output:List[torch.Tensor]):
		grad_output = grad_output.detach()
		input,lognorm,axis,totalcount = ctx.saved_tensors
		samps = sample(input-lognorm,axis,totalcount.item())
		#prob = (input - lognorm).exp()
		#samps = LogSumExpStoch.sample(prob,axis.item())
		grad_input = grad_output * samps
		return grad_input,None,None

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
		samples = sample(input,axis,axis * 0 + 1)
		ctx.save_for_backward(input,samples,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return samples
	@staticmethod
	def backward(ctx, grad_output:List[torch.Tensor]):
		#grad_output = grad_output.detach()
		input, samples, axis = ctx.saved_tensors
		inputp = input.exp()
		#samplesnew = Sampler.sample(inputp, axis)
		return grad_output*samples, None

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
class ExpStoch(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis):
		input = input.detach()
		inputp = input.exp()
		ctx.save_for_backward(inputp,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return inputp
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		#grad_output = grad_output.detach()

		inputp, axis = ctx.saved_tensors
		samplesnew = Sampler.sample(inputp, axis)
		grad_input = grad_output * samplesnew
		print(type(grad_output))
		return grad_input, None

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





