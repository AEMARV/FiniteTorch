from torch import Tensor
from layers.klfunctions import *
import torch.nn.functional as F
import torch.nn as nn
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
		if isuniform and not(isstoch):
			raise(Warning('Uniform Parameterizaiton, while being not stochastic'))
		self.isuniform = isuniform

	def __call__(self, shape:Tuple)->Tensor:
		if self.isbinary:
			self.normaxis=4
			shape = shape + (2,)
		out = torch.empty(shape)
		out = out.exponential_()
		if self.isuniform:
			out = out * 0
		return out
	def get_log_norm(self,k:Tensor)->Tensor:
		lognorm = self.normfunc(k,self.normaxis)
		return lognorm
	def get_log_kernel(self,k:Tensor)->Tuple[Tensor]:
		k =  k- self.get_log_norm(k)
		if self.isbinary:
			return k[0:,0:,0:,0:,0], k[0:,0:,0:,0:,1]
		else:
			return k
	def get_prob_kernel(self,k:Tensor)->Tensor:
		if self.isbinary:
			k0,k1 = self.get_log_kernel(k)
			k0 = k0.exp()
			k1 = k1.exp()
			return k0,k1
		else:
			k = self.get_log_kernel(k)
			k = k.exp()
			return k








