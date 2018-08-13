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
import klconvs
import stochgrads
import jconv
import argparse
def sample(lp:Tensor,axis:int,numsamples):
	lastaxis = lp.ndimension() -1
	lpt = lp.transpose(lastaxis,axis)
	M = Multinomial(total_count=numsamples,logits=lpt)
	#D = Dirichlet((lp.exp())*(numsamples.float())/(lp.size(lastaxis)))
	samps = M.sample()
	samps = samps.transpose(lastaxis,axis)/numsamples
	logprob = (samps * lp).sum(dim=axis,keepdim=True)
	return samps, logprob
def sampleprob(p:Tensor,axis:int,numsamples):
	lastaxis = p.ndimension() - 1
	p = p.transpose(lastaxis, axis)
	M = Multinomial(total_count=numsamples, probs=p)
	# D = Dirichlet((lp.exp())*(numsamples.float())/(lp.size(lastaxis)))
	samps = M.sample()
	samps = samps.transpose(lastaxis, axis) / numsamples
	return samps
class KLConvStoch(Function):
	@staticmethod
	def normal_klconv(input,log_filt:Tensor):
		p_filt = log_filt.exp()
		ker_size = log_filt.size()[3]
		pad = int((ker_size-1)/2)
		cross = F.conv2d(input, p_filt, padding=pad,stride=1,bias=None)

		H = p_filt * log_filt
		H = -H.sum(dim=1, keepdim=True)

		out = cross + F.conv2d(input[0:,0:1,0:,0:]*0 + 1, H, padding = pad, stride=1,bias=None)
		return out

	@staticmethod
	def forward(ctx, input, log_filt):
		input.detach()
		log_filt.detach()

		#output = KLConvStoch.normal_klconv(input, log_filt)
		output,random = klconvs.forward(input, log_filt)
		ctx.save_for_backward(input, log_filt,random)
		return output

	@staticmethod
	def backward(ctx, grad_outputs):
		grad_outputs.detach()
		input, log_filt,random = ctx.saved_tensors
		dzdin , dzdlog_filt = klconvs.backward(grad_outputs,input,log_filt,random)
		#dzdin, dzdlog_filt = klconvs.backward_rand(grad_outputs, input, log_filt)
		return dzdin, dzdlog_filt
class mix(Function):
	@staticmethod
	def normal_mix(input,log_filt:Tensor):
		m = (input.max(dim=1, keepdim=True))[0]
		x = input - m
		x = x.exp()
		output = F.conv2d(x, log_filt.exp())
		output = (output).log() + m
		return output

	@staticmethod
	def forward(ctx, input, log_filt):
		input.detach()
		log_filt.detach()
		m = (input.max(dim=1, keepdim=True))[0]
		x = input - m
		x = x.exp()
		output = F.conv2d(x, log_filt.exp())
		output = (output).log() + m
		#output,random = klconvs.forward(input, log_filt)
		ctx.save_for_backward(input, log_filt, output)
		return output

	@staticmethod
	def backward(ctx, grad_outputs):
		grad_outputs.detach()
		input, log_filt , output = ctx.saved_tensors
		dzdin, dzdlog_filt = stochgrads.mixer_backward(grad_outputs, input, output, log_filt)
		return dzdin, dzdlog_filt
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
		samps,lprob = sample(input-lognorm,axis,totalcount.item())
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
		samples = sampleprob(input,axis,1)
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
		#inputp = input.exp()
		samples,logprob = sample(input,axis,1)
		ctx.save_for_backward(input,samples,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return samples,logprob
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor,grad_logprob):
		#grad_output = grad_output.detach()

		input, samples, axis = ctx.saved_tensors
		#samplesnew = Sampler.sample(inputp, axis,1)
		grad_input = grad_output * input.exp()
		return grad_logprob*samples + (samples*grad_output), None

	@staticmethod
	def sample32(p:Tensor,axis):
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
class MaxExp(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,axis):
		input = input.detach()
		input = input*1000
		#inputp = input.exp()
		output = (input - LogSumExp.apply(input,1)).exp()
		ctx.save_for_backward(input,output,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return output
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor,grad_logprob):
		#grad_output = grad_output.detach()

		input, samples, axis = ctx.saved_tensors
		#samplesnew = Sampler.sample(inputp, axis,1)
		grad_input = grad_output * input.exp()
		return grad_logprob*samples, None

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


'''Loss Functions'''
class LossExpectedError(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,target:Tensor):
		target = target.unsqueeze(1)
		input = input.detach() #type:Tensor
		probs = input.exp()
		target_one_hot = input.new_zeros([input.size()[0],input.size()[1]]) #type: Tensor
		target_one_hot.scatter_(1,target,1)
		acc = (probs * target_one_hot).sum(dim=1,keepdim=True)
		#acc = acc.mean(dim=0,keepdim=False)
		ctx.save_for_backward(probs, target_one_hot)
		loss= -(acc.log())
		return loss
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		grad_output = grad_output.detach()
		probs,target_one_hot = ctx.saved_tensors
		batchsz = probs.size()[0]
		grad_input = -((target_one_hot))*grad_output#*probs
		return grad_input, None
class LossCrossEntropy(Function):
	@staticmethod
	def forward(ctx, input: torch.Tensor, target: Tensor):
		target = target.unsqueeze(1)
		input = input.detach()  # type:Tensor
		target_one_hot = input.new_zeros([input.size()[0], input.size()[1]])  # type: Tensor
		target_one_hot.scatter_(1, target, 1)
		loss = -(input * target_one_hot).sum(dim=1, keepdim=True)
		return loss

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		grad_output = grad_output.detach()
		target_one_hot = ctx.saved_tensors
		grad_input = -((target_one_hot)) * grad_output
		return grad_input, None
class jointConv(Function):
	@staticmethod
	def forward(ctx, input: torch.Tensor, log_filt: Tensor):

		input = input.detach()  # type:Tensor
		log_filt = log_filt.detach() #type:Tensor
		out = jconv.forward(input,log_filt)
		ctx.save_for_backward(input, log_filt)
		return out

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		grad_output = grad_output.detach() #type: Tensor
		grad_output = grad_output.contiguous()
		input, log_filt= ctx.saved_tensors
		grad_input, grad_filt = jconv.backward(grad_output,input,log_filt)
		return grad_input, grad_filt




