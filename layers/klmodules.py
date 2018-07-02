from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from definition import epsilon
from layers.klfunctions import *
from typing import Tuple,List,Dict
class KLConv_Base(Module):
	def __init__(self,
	             fnum=None,
	             kersize=None,
	             isbiased=False,
	             inp_chan_sz=0,
	             isrelu=True,
	             biasinit=None,
	             padding=None,
	             stride=1,
	             isstoch=False
	             ):
		super(KLConv_Base,self).__init__()
		#TODO: Set isbinary switch in paraminit
		self.biasinit = biasinit
		self.padding = num_pad_from_symb_pad(padding,kersize)
		self.isbiased = isbiased
		self.isrelu = isrelu
		self.stride = stride
		self.isstoch= isstoch
		self.axisdim=-2
		self.log_prob = 0
		# Build
		kernel_shape = (fnum,)+(inp_chan_sz,)+(kersize,kersize)
		self.kernel = Parameter(data=self.paraminit(kernel_shape))
		self.register_parameter('kernel',self.kernel)
		if self.isbiased:
			self.bias = Parameter(data=biasinit(fnum))

	'''Kernel/Bias Getters'''
	def get_log_kernel(self)-> torch.Tensor:
		''' Get the kernel in the log domain'''

		return self.paraminit.get_log_kernel(self.kernel)

	def get_prob_kernel(self)-> torch.Tensor:

		return self.paraminit.get_prob_kernel(self.kernel)

	def get_log_bias(self)-> torch.Tensor:
		return self.biasinit.get_log_bias(self.bias)

	def get_prob_bias(self)-> torch.Tensor:
		return self.biasinit.get_prob_bias(self.bias)

	def convwrap(self,x:Tensor,w:Parameter):
		y = F.conv2d(x, w, bias=None,
		             stride=self.stride,
		             padding=self.padding)
		return y

	def add_ker_ent(self,y:torch.Tensor,x,pker,lker):
		H = self.ent_per_spat(pker,lker)
		H = self.convwrap(x[0:,0:1,0:,0:]*0 +1,H)
		return y + H

class KLConv(KLConv_Base):
	def __init__(self,
	             paraminit=None,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		self.paraminit = paraminit
		self.paraminit.isbinary = False # DO NOT Move these lines after super
		super(KLConv,self).__init__(**kwargs)
		self.axisdim= 1



	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H
	'''KL Conv Functions'''

	def kl_xl_kp(self,x:torch.Tensor):
		'''conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor'''

		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y = self.add_ker_ent(y,x,pkernel,lkernel)
		return y

	def kl_xp_kl(self,x):
		#TODO : not implemented yet
		raise(Exception('This KLD is not implemented!'))

	def forward(self, x:torch.Tensor):
		if self.training:
			y = KLConvStoch.apply(x,self.get_log_kernel())
		else:
			y=  self.kl_xl_kp(x)
		return y



class KLConvB(KLConv_Base):
	def __init__(self,
	             paraminit=None,
	             **kwargs
	             ):
		self.paraminit = paraminit
		self.paraminit.isbinary = True # DO NOT Move these lines after super
		super(KLConvB,self).__init__(**kwargs)
		self.axisdim=4

	def ent_per_spat(self,pker,lker):
		# Entropy Per Spatial Position
		lker0,lker1 = self.seperate_kernels(lker)
		pker0,pker1 = self.seperate_kernels(pker)
		H = (pker0*lker0) + (pker1*lker1)
		H = -H.sum(dim=1,keepdim=True)
		return H
	def seperate_kernels(self,k):
		return k[0:,0:,0:,0:,0], k[0:,0:,0:,0:,1]
	'''KL Conv Functions'''
	def cross_xl_kp(self,x:Tensor,k0:Tensor,k1:Tensor):
		xp1 = x
		xp0 = 1- x
		xp1.clamp_(epsilon,1)
		xp0.clamp_(epsilon,1)
		xlog1 = xp1.log()
		xlog0 = xp0.log()
		y0 = self.convwrap(xlog0,k0)
		y1 = self.convwrap(xlog1,k1)
		y = y0 + y1
		return y
	def kl_xl_kp(self,x:torch.Tensor):
		'''conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor'''
		#xclamped = x.clamp(epsilon,1-epsilon)
		if self.isstoch and self.training:
			lk = self.get_log_kernel()
			pk = lk.exp()
			pk = Sampler.apply(lk,4)
			lk = pk.clamp(epsilon,None).log()
			pk0,pk1 = self.seperate_kernels(pk)
			y = self.cross_xl_kp(x,pk0,pk1)
			y = self.add_ker_ent(y, x, pk,lk)
		else:
			lk =self.get_log_kernel()
			pk = lk.exp()
			pk0,pk1 = self.seperate_kernels(pk)
			y = self.cross_xl_kp(x,pk0,pk1)
			y = self.add_ker_ent(y,x,pk,lk)
		return y

	def kl_xp_kl(self,x):
		#TODO : not implemented yet
		raise(Exception('This KLD is not implemented!'))

	def forward(self, x:torch.Tensor):
		return self.kl_xl_kp(x)


class KLAvgPool(Module):
	def __init__(self,spsize,stride,pad,isstoch=True):
		super(KLAvgPool,self).__init__()
		self.spsize= spsize
		self.stride = stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
		self.isstoch = isstoch

	def forward(self, x:Tensor):
		einput = x.exp()
		out = F.avg_pool2d(einput,
		                   self.spsize,
		                   stride=self.stride,
		                   padding=self.pad,
		                   count_include_pad=False)
		out = out.clamp(epsilon,None)
		out = out.log()
		return out
class KLAvgPoolGL(Module):
	def __init__(self,spsize,stride,pad,isstoch=True):
		super(KLAvgPoolGL,self).__init__()
		self.spsize= spsize
		self.stride = stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
		self.isstoch = isstoch

	def forward(self, x:Tensor):
		einput = x.exp()
		out = F.avg_pool2d(einput,
		                   self.spsize,
		                   stride=self.stride,
		                   padding=self.pad,
		                   count_include_pad=False)
		out = out.clamp(epsilon,None)
		out = out.log()
		return out


class LNorm(Module):
	def __init__(self,isstoch):
		super(LNorm,self).__init__()
		self.isstoch= isstoch
		if not(self.isstoch):
			self.implicit_layer = nn.LogSoftmax(dim=1)
			self.add_module('logsoft',self.implicit_layer)
	def forward(self, x):
		if self.isstoch:
			m = LogSumExpStoch.apply(x, 1,1)
			out = x - m
		else:
			out = self.implicit_layer(x)

		return out

class Inp2Log(Module):
	def __init__(self):
		super(Inp2Log,self).__init__()
	def forward(self, x:Tensor):
		x= x.clamp(epsilon,None) # type: Tensor
		x = x/x.sum(dim=1,keepdim=True)
		out = x.log()

		return out

def num_pad_from_symb_pad(pad:str,ksize:int)->Tuple[int]:
	if pad=='same':
		rsize = ksize
		csize = ksize
		padr = (rsize-1)/2
		padc = (csize-1)/2
		return (padr,padc)
	elif pad=='valid':
		padr=0
		padc=0
		return (padr,padc)
	elif type(pad) is tuple:
		return pad
	else:
		raise(Exception('Padding is unknown--Pad:',pad))
