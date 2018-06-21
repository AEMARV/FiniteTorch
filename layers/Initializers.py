from torch import Tensor
from layers.klfunctions import *
import torch.nn.functional as F
import torch.nn as nn
from definition import epsilon
import torch
from typing import List,Tuple,Dict
''' Interface Class'''
class Parameterizer(object):
	def __init__(self,isstoch=True,isbinary=False,dtype=torch.float32):
		self.isstoch = isstoch
		self.isbinary = isbinary
		if self.isbinary:
			self.normaxis = 4
		else:
			self.normaxis = 1
		self.dtype = dtype
	def __call__(self, shape, dtype='float32', isbinary=False):
		raise("Outputs a ndarray with 5-th dim being the indepenent components")
	def get_log_kernel(self,k:Tensor):
		raise(Exception("This class is Abstract"))
	def get_prob_kernel(self,k:Tensor):
		raise (Exception("This class is Abstract"))
	def get_log_norm(self,k:Tensor):
		raise (Exception("This class is Abstract"))
	def get_prob_norm(self,k:Tensor):
		raise (Exception("This class is Abstract"))


''' Dirichlet Inits'''
class LogParameter(Parameterizer):
	def __init__(self,
	             isstoch=True,
	             isuniform=True,
	             isdirichlet=True,
	             isbinary=False,
	             **kwargs):
		''' isuniform sets all the distributions to uniform
		and is equivalent to initializing with zeros in RealCNNS
		Obviously it is advised to turn on isstoch switch when uniform is on
		'''
		super(LogParameter,self).__init__(**kwargs)
		if isstoch:
			self.normfunc = LogSumExpStoch.apply
		else:
			self.normfunc = LogSumExp.apply
		self.isuniform = isuniform
		self.isbinary = isbinary
		self.isdirichlet= isdirichlet
		#if isuniform and not(isstoch):
		#	raise(Warning('Uniform Parameterizaiton, while being not stochastic'))
		self.isuniform = isuniform

	def __call__(self, shape:Tuple)->Tensor:
		if self.isbinary:
			self.normaxis=4
			shape = shape + (2,)
		out = torch.empty(shape)
		out = out.exponential_()
		if self.isdirichlet:
			out = out / out.sum(dim=self.normaxis,keepdim=True)
			out = out.clamp(epsilon, 1)
			out = out.log()
		if self.isuniform:
			out = out * 0
		return out.detach_()
	def get_log_norm(self,k:Tensor)->Tensor:
		lognorm = self.normfunc(k,self.normaxis,100)
		return lognorm
	def get_log_kernel(self,k:Tensor)->Tuple[Tensor]:
		k =  k- self.get_log_norm(k)
		#return k[0:,0:,0:,0:,0], k[0:,0:,0:,0:,1]
		return k
	def get_prob_kernel(self,k:Tensor)->Tensor:
		k = self.get_log_kernel(k)
		k = k.exp()
		return k

class SphereParameter(Parameterizer):
	def __init__(self,
	             isstoch=True,
	             isuniform=True,
	             isdirichlet=True,
	             isbinary=False,
	             **kwargs):
		''' isuniform sets all the distributions to uniform
		and is equivalent to initializing with zeros in RealCNNS
		Obviously it is advised to turn on isstoch switch when uniform is on
		'''
		super(SphereParameter,self).__init__(**kwargs)
		if isstoch:
			self.normfunc = LogSumExpStoch.apply
		else:
			self.normfunc = LogSumExp.apply
		self.isuniform = isuniform
		self.isbinary = isbinary
		self.isdirichlet= isdirichlet
		#if isuniform and not(isstoch):
		#	raise(Warning('Uniform Parameterizaiton, while being not stochastic'))
		self.isuniform = isuniform

	def __call__(self, shape:Tuple)->Tensor:
		if self.isbinary:
			self.normaxis=4
			shape = shape + (2,)
		out = torch.empty(shape)
		out = out.normal_(0,1)
		if self.isuniform:
			out = (out * 0) +1
		out = out / ((out ** 2).sum(dim=self.normaxis, keepdim=True)).sqrt_()
		out = out.clamp(epsilon, 1)
		return out.detach_()
	def get_log_norm(self,k:Tensor)->Tensor:
		lognorm = self.normfunc(2*((k.abs().clamp(epsilon,None)).log()),self.normaxis,1)
		return lognorm
	def get_log_kernel(self,k:Tensor)->Tuple[Tensor]:
		k = 2*((k.abs().clamp(epsilon,None)).log()) - self.get_log_norm(k)
		#return k[0:,0:,0:,0:,0], k[0:,0:,0:,0:,1]
		return k
	def get_prob_kernel(self,k:Tensor)->Tensor:
		#k = self.get_log_kernel(k)
		#k = k.exp()
		p = k**2

		return p/(p.sum(dim=self.normaxis,keepdim=True))






