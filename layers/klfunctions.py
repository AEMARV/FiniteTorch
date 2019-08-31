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
from layers.pmaputils import *
import argparse
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

class md_conv(Function):
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
		output = mdconv.forward(input, log_filt)
		ctx.save_for_backward(input, log_filt)
		return output

	@staticmethod
	def backward(ctx, grad_outputs):
		grad_outputs.detach()
		input, log_filt = ctx.saved_tensors
		dzdin , dzdlog_filt = mdconv.backward(grad_outputs,input,log_filt)
		#dzdin, dzdlog_filt = klconvs.backward_rand(grad_outputs, input, log_filt)
		return dzdin, dzdlog_filt
class mdi_conv(Function):
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
		output = mdconv.iforward(input, log_filt)
		ctx.save_for_backward(input, log_filt)
		return output

	@staticmethod
	def backward(ctx, grad_outputs):
		grad_outputs.detach()
		input, log_filt = ctx.saved_tensors
		dzdin , dzdlog_filt = mdconv.ibackward(grad_outputs,input,log_filt)
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
		logsumexp = input.logsumexp(dim=axis,keepdim=True)
		ctx.save_for_backward((input-logsumexp).detach(),Variable(input.new_tensor(axis,dtype=torch.int32)),Variable(input.new_tensor(totalcount,dtype=torch.int32)))
		return logsumexp
	@staticmethod
	def backward(ctx, grad_output:List[torch.Tensor]):
		grad_output = grad_output.detach()
		input,axis,totalcount = ctx.saved_tensors

		#prob = (input - lognorm).exp()
		#samps = LogSumExpStoch.sample(prob,axis.item())
		if totalcount==1:
			samps = sample_manual(input, axis)[0]
			grad_input = grad_output * samps
		else:
			grad_input = input.exp()*grad_output
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
class Sampler_f(Function):
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
		samples,logprob = sample(input,axis,1)
		#samples= input.exp()
		ctx.save_for_backward(input,samples,Variable(input.new_tensor(axis,dtype=torch.int32)))
		return samples
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		#grad_output = grad_output.detach()

		input, samples, axis = ctx.saved_tensors
		return grad_output * samples, None

	@staticmethod
	def sampleMine(p:Tensor,axis):
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


class JSDivg(Function):
	@staticmethod
	def forward(ctx, input:Tensor, target:Tensor ):
		target = target.unsqueeze(1)
		target_one_hot = input.new_zeros([input.size()[0], input.size()[1]])  # type: Tensor
		target_one_hot.scatter_(1, target, 1)
		probs = input.exp()
		mixed = (probs + target_one_hot)/2
		divg = (mixed.log()-input) #type: Tensor
		divg,d = divg.min(dim=1,keepdim=True)
		return -divg


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

'''Probabilistic Loss'''
class PosteriorLoss(Function):
	''' Takes log sum exp along axis 1'''
	@staticmethod
	def forward(ctx, input:torch.Tensor,target:Tensor):
		raise Exception('suspicious')
		target = target.unsqueeze(1)
		input = input.detach() #type:Tensor
		probs = input.exp()
		target_one_hot = input.new_zeros([input.size()[0],input.size()[1]]) #type: Tensor
		target_one_hot.scatter_(1,target,1)
		acc = (probs * target_one_hot).sum(dim=1,keepdim=True)
		#acc = acc.mean(dim=0,keepdim=False)
		ctx.save_for_backward(probs, target_one_hot)
		loss = -(acc.log())
		return loss
	@staticmethod
	def backward(ctx, grad_output:torch.Tensor):
		grad_output = grad_output.detach()
		probs,target_one_hot = ctx.saved_tensors
		batchsz = probs.size()[0]

		zeros = probs.new_zeros([probs.size()[0], probs.size()[1]])  # type: Tensor
		target_one_hot_model = zeros.scatter(1,torch.argmax(probs,dim=1,keepdim=True),1)
		target_one_hot_model_g_oracle = zeros.scatter(1, torch.argmax(probs*(1-target_one_hot), dim=1, keepdim=True), 1)

		grad_oracle_g_model = (-target_one_hot) * grad_output   #*(probs)
		grad_model = target_one_hot_model*grad_output
		grad_model_g_oracle = target_one_hot_model_g_oracle* grad_output

		grad_input = grad_oracle_g_model# - grad_model
		return grad_input, None



'''PMAPS'''
class ConvBayesMap(Function):
	@staticmethod
	def forwardval(ctx, lx, lkernel,lbias, trials,pad):
		lx.detach_()
		lkernel.detach_()
		lbias.detach_()
		ynorm = 0
		for i in range(trials):
			xsample,l = sample(lx,1,1)
			y = F.conv2d(xsample,lkernel,padding=int(pad)) + lbias
			ynormtemp= y - LogSumExp.apply(y,1) #type: Tensor
			ynorm += ynormtemp.exp()
		ynormall = ynorm.log()
		ynorm = ynorm/trials

		ynorm = ynorm.log()
		ctx.save_for_backward(lx,lkernel,lbias, xsample,ynormtemp,ynormall, Variable(lx.new_tensor(trials,dtype=torch.int32)),Variable(lx.new_tensor(pad,dtype=torch.int32)))
		return ynorm

	@staticmethod
	def forward(ctx, lx, lkernel, lbias, trials, pad,stride,option):
		lx = lx.detach()
		lkernel = lkernel.detach()
		lbias = lbias.detach()

		ynorm = 0

		log_ynorm,xsample = ConvBayesMap.singleforward(lx,lkernel,lbias,pad,stride)



		ctx.save_for_backward(lx, lkernel, lbias, log_ynorm,xsample,
		                      Variable(lx.new_tensor(trials, dtype=torch.int32)),
		                      Variable(lx.new_tensor(pad, dtype=torch.int32)),
		                      Variable(lx.new_tensor(stride, dtype=torch.int32)),
		                      Variable(lx.new_tensor(option, dtype=torch.int32))
		                      )
		return log_ynorm
	@staticmethod
	def singleforward(lx,lkernel,lbias,pad,stride):
		xsample = sample(lx, 1, 1)[0]
		y = F.conv2d(xsample, lkernel, padding=int(pad),stride=stride) + lbias
		normalizer = LogSumExp.apply(y, 1)
		ynorm = y -  normalizer # type: Tensor
		#print("norm:{:.2E}".format(normalizer.exp().mean().item()),end=" ")
		return ynorm,xsample

	@staticmethod
	def backward(ctx,dzdynorm):
		lx, lkernel, lbias, logynorm, xsample, trials, pad, stride,option = ctx.saved_tensors
		option = option.item()
		if option ==0:
			grads = ConvBayesMap.backward_rejection(lx,lkernel,lbias,pad,stride,dzdynorm)
		elif option ==1:
			grads =  ConvBayesMap.backward_mile(lx,lkernel,lbias,pad,stride,dzdynorm)
		elif option ==2:
			grads =  ConvBayesMap.backward_singlesample(lx,lkernel,lbias,logynorm,xsample,pad,stride,dzdynorm)
		return grads+(None,None,None,None)
	@staticmethod
	def backward_mile(lx,lkernel,lbias,pad,stride, dzdynorm):
		'''
		Mile Sampling
		:param ctx:
		:param dzdynorm:
		:return:
		'''
		dzdynorm = dzdynorm.detach()
		pad = pad.item()
		stride = stride.item()
		r = dzdynorm.abs()
		dzdlx= 0
		dzdlker = 0
		dzdlbias = 0
		totalenergy = r.sum()
		#for i in range(1):
		while r.sum()>totalenergy*0.00:

			#print(r.mean().item(),end="\r")
			ynorm_temp, xsample_temp = ConvBayesMap.singleforward(lx,lkernel,lbias,pad,stride)
			coef = torch.min(r,ynorm_temp.exp())
			#coef = r * ynorm_temp.exp()
			r = r - coef
			dzdlx1,dzdlkernel1,dzdlbias1 = p_map_backwardv1(lx,lkernel,lbias,xsample_temp,ynorm_temp,pad,stride, coef*(dzdynorm.sign()))

			dzdlx += dzdlx1
			dzdlker += dzdlkernel1
			dzdlbias += dzdlbias1
			#print((r).sum().item())
			if r.sum()==0:
				break
		if False:

			lxmean = (ynorm_temp.exp().mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) + 1e-7).log()
			avgentropy = (-(ynorm_temp * ynorm_temp.exp()).sum(dim=1)).mean() / math.log(float(ynorm_temp.shape[1]))
			info = (-lxmean * lxmean.exp()).sum() / math.log(float(ynorm_temp.shape[1]))
			print("info:{:.2f}".format(info.item()), end=",")
			print("ent:{:.2f}".format(avgentropy.item()), end="||")
		#print((dzdlker).abs().mean().item())

		return dzdlx, dzdlker, dzdlbias

	@staticmethod
	def backward_singlesample(lx, lkernel, lbias,logynorm,xsample, pad, stride, dzdynorm):
		''' Rejection sampling'''
		dzdynorm = dzdynorm.detach()

		pad = pad.item()
		stride = stride.item()
		dzdlx = 0
		dzdlker = 0
		dzdlbias = 0
		ysample = sample(logynorm,1,1)[0]
		dzdlx1, dzdlkernel1, dzdlbias1 = p_map_backwardv1(lx, lkernel, lbias, xsample, logynorm, pad,
		                                                  stride, dzdynorm)
		dzdlx += dzdlx1
		dzdlker += dzdlkernel1
		dzdlbias += dzdlbias1
		if False:

			lxmean = (logynorm.exp().mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) + 1e-7).log()
			avgentropy = (-(logynorm* logynorm.exp()).sum(dim=1)).mean() / math.log(float(logynorm.shape[1]))
			info = (-lxmean * lxmean.exp()).sum() / math.log(float(logynorm.shape[1]))
			print("info:{:.2f}".format(info.item()), end=",")
			print("ent:{:.2f}".format(avgentropy.item()), end=" ")
		return dzdlx, dzdlker, dzdlbias
	@staticmethod
	def backward_exp(ctx, dzdynorm):
		''' Rejection sampling'''
		dzdynorm = dzdynorm.detach()
		lx, lkernel, lbias, log_ynorm, xsample, trials, pad, stride = ctx.saved_tensors
		ynorm=  log_ynorm.exp()
		pad = pad.item()
		stride = stride.item()
		dzdlx = 0
		dzdlker = 0
		dzdlbias = 0
		dzdlx1, dzdlkernel1, dzdlbias1 = p_map_backwardv1(lx, lkernel, lbias, xsample, log_ynorm, trials, pad,
		                                                  stride, dzdynorm)
		dzdlx += dzdlx1
		dzdlker += dzdlkernel1
		dzdlbias += dzdlbias1
		# print((r).sum().item())
		# dzdlx2, dzdlkernel2, dzdlbias2 = p_map_backwardv1(lx, lkernel, lbias, xsample2, log_ynorm2, trials, pad,stride,
		#                                                  dzdynorm *rate2)

		if False:
			lxmean = (ynorm.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3,
			                                                                                    keepdim=True) + 1e-7).log()
			avgentropy = (-(ynorm * log_ynorm).sum(dim=1)).mean() / math.log(float(ynorm.shape[1]))
			info = (-lxmean * lxmean.exp()).sum() / math.log(float(ynorm.shape[1]))
			print("info:{:.2f}".format(info.item()), end=",")
			print("ent:{:.2f}".format(avgentropy.item()), end="||")
		# print((dzdlker).abs().mean().item())

		return dzdlx, dzdlker, dzdlbias, None, None, None
	@staticmethod
	def backward_rejection(lx,lkernel,lbias,pad,stride, dzdynorm):
		''' Rejection sampling'''
		dzdynorm = dzdynorm.detach()

		pad = pad.item()
		stride = stride.item()
		remain = dzdynorm

		dzdlx= 0
		dzdlker = 0
		dzdlbias = 0
		while remain.abs().sum()>0:
			#random = torch.rand_like(dzdynorm)

			ynorm_temp, xsample_temp = ConvBayesMap.singleforward(lx,lkernel,lbias,pad,stride)
			#isaccept = (random <= ynorm_temp.exp()).float()
			isaccept = (sample(ynorm_temp,1,1)[0]).float()
			#print((remain.abs()*(1-isaccept)).max()[0])
			#coef = r * ynorm_temp.exp()
			dzdynorm = (isaccept) * remain
			remain = remain - (isaccept)*remain
			#dzdlx1,dzdlkernel1,dzdlbias1 = p_map_backwardv1(lx,lkernel,lbias,xsample_temp,ynorm_temp,trials,pad,stride, coef*(dzdynorm.sign()))
			dzdlx1, dzdlkernel1, dzdlbias1 = p_map_backwardv1(lx, lkernel, lbias, xsample_temp, ynorm_temp, pad,
			                                                  stride, dzdynorm)
			dzdlx += dzdlx1
			dzdlker += dzdlkernel1
			dzdlbias += dzdlbias1
			#print((r).sum().item())
		#dzdlx2, dzdlkernel2, dzdlbias2 = p_map_backwardv1(lx, lkernel, lbias, xsample2, log_ynorm2, trials, pad,stride,
		#                                                  dzdynorm *rate2)

		if False:

			lxmean = (ynorm_temp.exp().mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) + 1e-7).log()
			avgentropy = (-(ynorm_temp * ynorm_temp.exp()).sum(dim=1)).mean() / math.log(float(ynorm_temp.shape[1]))
			info = (-lxmean * lxmean.exp()).sum() / math.log(float(ynorm_temp.shape[1]))
			print("info:{:.2f}".format(info.item()), end=",")
			print("ent:{:.2f}".format(avgentropy.item()), end="||")
		#print((dzdlker).abs().mean().item())

		return dzdlx, dzdlker, dzdlbias
	@staticmethod
	def backward_attempt_on_using_rejects(ctx, dzdynorm):
		''' Rejection sampling_using Rejects'''
		dzdynorm = dzdynorm.detach()
		lx, lkernel, lbias,trials,pad,stride = ctx.saved_tensors

		pad = pad.item()
		stride = stride.item()
		remain = dzdynorm

		dzdlx= 0
		dzdlker = 0
		dzdlbias = 0
		totalenergy = remain.abs().sum()
		isaccept = False
		accepted = False
		for i in range(1):
		#while remain.abs().sum()>totalenergy*0.00:
			random = torch.rand_like(dzdynorm)

			ynorm_temp, xsample_temp = ConvBayesMap.singleforward(lx,lkernel,lbias,pad,stride)
			ynorm_px, xsample_px = ConvBayesMap.singleforward(lx, lkernel, lbias, pad, stride)
			isaccept = (random <= ynorm_temp.exp()).float()
			#print(random)
			#coef = r * ynorm_temp.exp()
			dzdynorm_accept = (isaccept) * dzdynorm

			#dzdlx1,dzdlkernel1,dzdlbias1 = p_map_backwardv1(lx,lkernel,lbias,xsample_temp,ynorm_temp,trials,pad,stride, coef*(dzdynorm.sign()))
			dzdlx1, dzdlkernel1, dzdlbias1 = p_map_backwardv1(lx, lkernel, lbias, xsample_temp, ynorm_temp, trials, pad,
			                                                  stride, dzdynorm_accept)
			dzdlx += dzdlx1
			dzdlker += dzdlkernel1
			dzdlbias += dzdlbias1
			dzdlx1, dzdlkernel1, dzdlbias1 = p_map_backwardv1(lx, lkernel, lbias, xsample_temp, ynorm_temp, trials, pad,
		                                                  stride, dzdynorm*(1-isaccept))

			dzdlx_px, dzdlkernel_px, dzdlbias_px = p_map_backwardv1(lx, lkernel, lbias, xsample_px, ynorm_px, trials, pad,
			                                                  stride, dzdynorm*(1- isaccept))
			dzdlx += (dzdlx_px - dzdlx1)
			dzdlker += (dzdlkernel_px -  dzdlkernel1)
			dzdlbias += (dzdlbias_px - dzdlbias1)
			#print((r).sum().item())
		#dzdlx2, dzdlkernel2, dzdlbias2 = p_map_backwardv1(lx, lkernel, lbias, xsample2, log_ynorm2, trials, pad,stride,
		#                                                  dzdynorm *rate2)

		if True:

			lxmean = (ynorm_temp.exp().mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) + 1e-7).log()
			avgentropy = (-(ynorm_temp * ynorm_temp.exp()).sum(dim=1)).mean() / math.log(float(ynorm_temp.shape[1]))
			info = (-lxmean * lxmean.exp()).sum() / math.log(float(ynorm_temp.shape[1]))
			print("info:{:.2f}".format(info.item()), end=",")
			print("ent:{:.2f}".format(avgentropy.item()), end="||")
		#print((dzdlker).abs().mean().item())

		return dzdlx, dzdlker, dzdlbias, None, None,None
	@staticmethod
	def backward_func(lx, lkernel,lbias,trials,pad,stride, dzdynorm):
		dzdynorm = dzdynorm.detach()

		pad = pad.item()
		stride = stride.item()
		r = dzdynorm.abs()
		dzdlx= 0
		dzdlker = 0
		dzdlbias = 0
		for i in range(1):
		#while r.sum()>0:
		#	print(r.mean().item(),end="\r")
			ynorm_temp, xsample_temp = ConvBayesMap.singleforward(lx,lkernel,lbias,pad,stride)
			coef = torch.min(r,ynorm_temp.exp())
			r = r - coef
			dzdlx1,dzdlkernel1,dzdlbias1 = p_map_backwardv1(lx,lkernel,lbias,xsample_temp,ynorm_temp,trials,pad,stride, coef*dzdynorm.sign())
			dzdlx += dzdlx1
			dzdlker += dzdlkernel1
			dzdlbias += dzdlbias1
			if r.sum() < 0.001:
				break
			#print((r).sum().item())
		#dzdlx2, dzdlkernel2, dzdlbias2 = p_map_backwardv1(lx, lkernel, lbias, xsample2, log_ynorm2, trials, pad,stride,
		#                                                  dzdynorm *rate2)
		if False:
			lxmean = (lx.exp().mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) + 1e-7).log()
			entropy = (-lxmean * lxmean.exp()).sum() / math.log(float(lx.shape[1]))
			print("ent:{:.2f}".format((-(lx * lx.exp()).sum(dim=1)).mean() / math.log(float(lx.shape[1]))), end=",")
			print("info:{:.2f}".format(entropy), end="||")
		return dzdlx, dzdlker, dzdlbias, None, None,None

	def backward_legacy(ctx, dzdynorm):
		dzdynorm.detach_()
		lx, lkernel, lbias, sample,ynorm ,trials,pad = ctx.saved_tensors
		x = lx.exp()
		kernel = lkernel.exp()
		x.detach_()
		kernel.detach_()
		pad = pad.item()
		with torch.enable_grad():
			kernel.requires_grad = True
			sample.requires_grad = True
			ytemp = F.conv2d(sample,kernel,padding=pad)
			dzdlx, dzdlkernel = torch.autograd.grad(ytemp,[sample,kernel],grad_outputs=dzdynorm,only_inputs=True)
		return dzdlx, dzdlkernel, lbias*0, None, None


class identity(Function):
	@staticmethod
	def forward(ctx, lx):
		samp = sample(lx,1,1)[0].detach()
		ctx.save_for_backward(samp)
		return samp.log()
	@staticmethod
	def backward(ctx, dzdly):
		dzdly = dzdly.detach()
		samp, = ctx.saved_tensors
		dzdlx = samp * dzdly
		#dzdlx = -(dzdly.abs().sum(dim=(1,2,3),keepdim=True)!=0).float() * samp
		return dzdlx



class CovBayesMap_I(Function):
	@staticmethod
	def forward(ctx, lx, lkernel, lbias, trials, pad,stride):
		lx = lx.detach()
		lkernel = lkernel.detach()
		lbias = lbias.detach()

		ynorm = 0
		for i in range(trials):
			log_ynorm,xsample = ConvBayesMap.singleforward_i(lx,lkernel,lbias,pad,stride)
			ynorm = (ynorm + log_ynorm.exp())
		ynorm = ynorm / trials
		log_ynorm = ynorm.log()


		ctx.save_for_backward(lx, lkernel, lbias,
		                      Variable(lx.new_tensor(trials, dtype=torch.int32)),
		                      Variable(lx.new_tensor(pad, dtype=torch.int32)),
		                      Variable(lx.new_tensor(stride, dtype=torch.int32)))
		return log_ynorm



	@staticmethod
	def singleforward_i(lx,lkernel,lbias,pad,stride):
		kersample, l = sample(lkernel,1,1)
		y = F.conv2d(lx,kersample,padding=int(pad),stride=stride)
		normalizer = LogSumExp.apply(y, 1)
		ynorm = y - normalizer  # type: Tensor
		return ynorm,kersample

'''Pooling'''

class glavgpool(Function):
	@staticmethod
	def forward(ctx, x):
		out = x.exp().mean(dim=(2,3),keepdim=True)
		#out = out.mean(dim=3, keepdim=True)
		out = out.clamp(definition.epsilon,None)
		out = out.log()
		return out,torch.zeros(1,1,dtype=out.dtype)
