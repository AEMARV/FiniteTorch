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
	             ):
		super(KLConv_Base,self).__init__()
		#TODO: Set isbinary switch in paraminit
		self.biasinit = biasinit
		self.padding = num_pad_from_symb_pad(padding,kersize)
		self.isbiased = isbiased
		self.isrelu = isrelu
		self.stride = stride
		# Build
		kernel_shape = (fnum,)+(inp_chan_sz,)+(kersize,kersize)
		self.kernel = Parameter(data=self.paraminit(kernel_shape))
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
	def add_ker_ent(self,y:torch.Tensor,x):
		H = self.ent_per_spat()
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



	def ent_per_spat(self):
		# Entropy Per spatial Position
		H = self.get_log_kernel()* self.get_prob_kernel()
		H = -H.sum(dim=1,keepdim=True)
		return H
	'''KL Conv Functions'''

	def kl_xl_kp(self,x:torch.Tensor):
		'''conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor'''
		pkernel = self.get_prob_kernel()
		y = self.convwrap(x,pkernel)
		y = self.add_ker_ent(y,x)
		return y

	def kl_xp_kl(self,x):
		#TODO : not implemented yet
		raise(Exception('This KLD is not implemented!'))

	def forward(self, x:torch.Tensor):
		return self.kl_xl_kp(x)



class KLConvB(KLConv_Base):
	def __init__(self,
	             paraminit=None,
	             **kwargs
	             ):
		self.paraminit = paraminit
		self.paraminit.isbinary = True # DO NOT Move these lines after super
		super(KLConvB,self).__init__(**kwargs)

	def ent_per_spat(self):
		# Entropy Per Spatial Position
		lker0,lker1 = self.get_log_kernel()
		pker0,pker1 = self.get_prob_kernel()
		H = (pker0*lker0) + (pker1*lker1)
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''KL Conv Functions'''

	def kl_xl_kp(self,x:torch.Tensor):
		'''conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor'''
		xclamped = x.clamp(epsilon,1-epsilon)
		xlog1 = xclamped.log()
		xlog0 = (1-xclamped).log()
		pkernel0,pkernel1 = self.get_prob_kernel()
		y0 = self.convwrap(xlog0,pkernel0)
		y1 = self.convwrap(xlog1,pkernel1)
		y = y0 + y1
		y = self.add_ker_ent(y,x)
		return y

	def kl_xp_kl(self,x):
		#TODO : not implemented yet
		raise(Exception('This KLD is not implemented!'))

	def forward(self, x:torch.Tensor):
		return self.kl_xl_kp(x)


class KLAvgPool(Module):
	def __init__(self,spsize,stride,pad):
		super(KLAvgPool,self).__init__()
		self.spsize= spsize
		self.stride = stride
		self.pad = num_pad_from_symb_pad(pad,spsize)

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
	def forward(self, x):
		if self.isstoch:
			m = LogSumExpStoch.apply(x, 1)
		else:
			m = LogSumExp.apply(x, 1)
		out = x - m
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
