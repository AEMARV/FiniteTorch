import numpy as np
import torch
import torch.functional
from torch.autograd import Function,Variable
import torch.nn
import scipy.special as sc
import torch.nn.functional as F
from torch import Tensor
import argparse

class LogSumExpStoch(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis=1):
		output = F.log_softmax(input,dim=axis)
		ctx.save_for_backward(input,output,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return output
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		#grad_output = grad_output.detach().numpy()
		input,lognorm,axis = ctx.saved_tensors
		prob = (input - lognorm).exp()
		samps = LogSumExpStoch.sample(prob,axis.item())
		grad_input = grad_output * samps
		return grad_input

	@staticmethod
	def sample(p:Tensor,axis):
		cumprob = p.cumsum(axis)
		sh = p.size()
		if axis==1 :
			rand = p.new_zeros(sh[0],1,sh[2],sh[3]).uniform_(0,1)
			#np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
		else:
			rand = p.new_zeros(sh[0], sh[1], sh[2], sh[3],1).uniform_(0,1)
			#np_rand = np.random.uniform(low=0.0, high=1.0, size=(sh[0], sh[1], sh[2], sh[3],1))
		samps = cumprob > rand
		if axis == 1:
			samps[0:, 1:, 0:, 0:] = samps[0:, 1:, 0:, 0:] ^ samps[0:,0:-1,0:,0:]
		elif axis == 5:
			samps[0:, 0:, 0:, 0:, 1:] = samps[0:, 0:, 0:, 0:, 1:] ^ samps[0:,0:,0:,0:,0:-1]
		else:
			raise(Exception('Only axis=1 and axis=5(for binary distributions) is acceptable '+ 'axis=%d'%axis+ ' was given'))
		samps = samps.type_as(p)
		return samps

class LogSumExp(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis=1):
		np_input = input.detach().numpy()
		np_output = sc.logsumexp(np_input,axis=axis,keepdims=True)
		ctx.save_for_backward(input,np_output,axis)
		return input.new(np_input)
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		grad_output = grad_output.detach().numpy()
		input,lognorm,axis = ctx.saved_tensors
		prob = np.exp(input - lognorm)
		grad_input = grad_output * prob
		return grad_output.new_tensor(grad_input)

	@staticmethod
	def sample(p:np.ndarray,axis=1):
		cumprob = np.cumsum(p,axis=axis)
		sh = np.shape(cumprob)
		np_rand = np.random.uniform(low=0.0,high=1.0,size=(sh[0],1,sh[2],sh[3]))
		samps = cumprob > np_rand

		return samps




