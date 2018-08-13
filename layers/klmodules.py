from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from definition import epsilon
from layers.klfunctions import *
from typing import Tuple,List,Dict
from layers.Initializers import *
import math
class MyModule(Module):
	def __init__(self,*args,blockidx=None,**kwargs):
		super(MyModule,self).__init__(*args,**kwargs)
		self.logprob = torch.zeros(1).to('cuda:0')
		self.regularizer = torch.zeros(1).to('cuda:0')
		self.scalar_dict = {}
		if blockidx is None:
			raise Exception('blockidx is None')
		self.blockidx = blockidx
		self.compact_name = type(self).__name__ + '({})'.format(blockidx)
		self.register_forward_hook(self.update_scalar_dict)
	def update_scalar_dict(self,self2,input,output):
		return
	def get_log_prob(self):
		lprob  = self.logprob
		for m in self.children():
			if isinstance(m,MyModule):
				lprob = lprob + m.get_log_prob()
		self.logprob = 0
		return lprob
	def get_reg_vals(self):
		reg = self.regularizer
		for m in self.children():
			if isinstance(m, MyModule):
				reg = reg + m.get_reg_vals()
		self.regularizer = 0
		return reg

	def get_scalar_dict(self):

		for m in self.children():
			if isinstance(m, MyModule):
				self.scalar_dict.update(m.get_scalar_dict())

		return self.scalar_dict


class KLConv_Base(MyModule):
	def __init__(self,
	             *args,
	             fnum=None,
	             kersize=None,
	             isbiased=False,
	             inp_chan_sz=0,
	             isrelu=True,
	             biasinit=None,
	             padding='same',
	             stride=1,
	             paraminit= None,
	             coefinit = 1,
	             isstoch=False,
	             **kwargs
	             ):
		super(KLConv_Base,self).__init__(*args,**kwargs)
		#TODO: Set isbinary switch in paraminit
		self.biasinit = biasinit
		self.padding = num_pad_from_symb_pad(padding,kersize)
		self.isbiased = isbiased
		self.isrelu = isrelu
		self.stride = stride
		self.isstoch= isstoch
		self.axisdim=-2
		self.coefinit = coefinit
		self.kernel_shape = (fnum,)+(inp_chan_sz,)+(kersize,kersize)
		self.paraminit = paraminit
		self.input_is_binary= False
	'''Scalar Measurements'''
	def get_scalar_dict(self):
		#y = self.scalar_dict.copy()
		#self.scalar_dict = {}
		return self.scalar_dict
	def update_scalar_dict(self,self2,input,output):
		#Filter Entorpy
		if type(input) is tuple:
			input = input[0]

		temp_dict = {self.compact_name +'| Kernel Entropy' : self.expctd_ent_kernel().item(),
		             self.compact_name +'| Input Entropy' : self.expctd_ent_input(input).item(),
		             }
		for key in temp_dict.keys():
			if key in self.scalar_dict:
				self.scalar_dict[key] = self.scalar_dict[key]/2 + temp_dict[key]/2
			else:
				self.scalar_dict[key] = temp_dict[key]

	''' Build'''

	def build(self):
		self.paraminit.coef = self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		self.register_parameter('kernel', self.kernel)
		if self.isbiased:
			self.bias = Parameter(data=self.paraminit((1,self.kernel_shape[0],1,1)))
			self.register_parameter('bias', self.bias)
	'''Kernel/Bias Getters'''
	def get_log_kernel(self)-> torch.Tensor:
		''' Get the kernel in the log domain'''

		return self.paraminit.get_log_kernel(self.kernel)

	def get_prob_kernel(self)-> torch.Tensor:

		return self.paraminit.get_prob_kernel(self.kernel)

	def get_log_bias(self):
		if self.isbiased:
			return self.biasinit.get_log_kernel(self.bias)
		else:
			return - math.log(self.kernel_shape[0])

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
		hall = H.mean()
		return y + H,hall

	def get_log_prob(self):
		lprob = self.logprob
		self.logprob= 0
		return lprob
	'''Entorpy Functions'''
	def ent_per_spat(self):
		raise Exception("Implement this function")
	def ent_kernel(self):
		return self.ent_per_spat(self.get_prob_kernel(),self.get_log_kernel()).sum()
	def expctd_ent_kernel(self):
		return self.ent_per_spat(self.get_prob_kernel(),self.get_log_kernel()).mean()/self.get_log_symbols()
	def ent_input_per_spat(self,x):
		ent = -x * x.exp()
		ent = ent.sum(dim=1, keepdim=True)
		return ent
	def expctd_ent_input(self,x):
		ent = self.ent_input_per_spat(x)
		return ent.mean()/self.get_log_symbols()
	def ent_input(self,x):
		ent = self.ent_input_per_spat(x)#type:Tensor
		e = ent.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return e
	def get_log_symbols(self):
		if self.input_is_binary:
			syms = self.kernel_shape[1]*math.log(2)
		else:
			syms = math.log(self.kernel_shape[1])
		return syms






class KLConv(KLConv_Base):
	def __init__(self,*args,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(KLConv,self).__init__(*args,**kwargs)
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.build()


	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''

	def kl_xl_kp(self,x:torch.Tensor):
		''' Relu KL Conv '''
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y,x,pkernel,lkernel)
		return y,ent
	def cross_xl_kp(self,x):
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x, pkernel)
		g, ent = self.add_ker_ent(y, x, pkernel, lkernel)
		return y, ent

	def kl_xp_kl(self,xl):
		#TODO : not implemented yet
		xp = xl.exp()
		ent = -xp * xl #type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp,self.get_log_kernel())
		ent = self.convwrap(ent,self.get_prob_kernel()[0:1,0:1,0:,0:]*0 + 1)
		y = cross + ent
		ent = ent.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return y,ent

	def kl_xp_kp(self,xl):
		xp = xl.exp()
		kp = self.get_log_kernel().exp()
		y = self.convwrap(xp,kp)
		return y.log()

	def kl_xp_kl_stoch(self,xl):
		''' Stoch Sigmoid KL CONV'''
		if self.training:
			xp,lprob = ExpStoch.apply(xl,1)
			self.logprob = lprob.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
		else:
			xp = MaxExp.apply(xl,1)

		#xl = xp.clamp(epsilon,None).log()
		ent = -xp * xl #type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp,self.get_log_kernel())
		#ent = self.convwrap(ent,self.get_prob_kernel()[0:1,0:1,0:,0:]*0 + 1)
		return cross# + ent

	def cross_xp_kl(self,xl):
		xp = xl.exp()
		ent = -xp * xl  # type: Tensor
		ent = ent.sum(dim=1, keepdim=True)
		cross = self.convwrap(xp, self.get_log_kernel())
		ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)
		ent = ent.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return cross,ent

	def cross_xp_kl_stoch(self,xl):
		xp, lprob = ExpStoch.apply(xl,1)
		self.logprob= lprob.sum(dim=1,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)
		return self.convwrap(xp,self.get_log_kernel())

	def forward(self, x:torch.Tensor):

		if not self.isrelu:
			# Sigmoid Activated
			if self.training and self.isstoch:
				y = self.kl_xp_kl_stoch(x)
			else:
				y,g = self.kl_xp_kl(x)
				self.logprob = self.ent_kernel() / 1000
				# ent = self.ent_input_per_spat(x).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1,
				#                                                                                         keepdim=False)
				# self.logprob = ent
		else:
			# ReLU Activated
			if self.training and self.isstoch :
				y = KLConvStoch.apply(x,self.get_log_kernel())
			else:
				y,ent = self.kl_xl_kp(x)


		y = y + self.get_log_bias()

		return y


class KLConvB(KLConv_Base):
	def __init__(self,
	             *args,
	             **kwargs
	             ):
		super(KLConvB,self).__init__(*args,**kwargs)
		self.paraminit.isbinary = True # DO NOT Move these lines after super
		self.axisdim=4
		self.input_is_binary = True
		self.build()

	def ent_per_spat(self,pker,lker):
		# Entropy Per Spatial Position
		lker0,lker1 = self.seperate_kernels(lker)
		pker0,pker1 = self.seperate_kernels(pker)
		H = (pker0*lker0) + (pker1*lker1)
		H = -H.sum(dim=1,keepdim=True)
		return H

	def ent_input_per_spat(self,x:Tensor):
		xl = x.clamp(epsilon,None).log()
		ent =-x*xl
		ent = ent.sum(dim=1,keepdim=True)
		return ent
	def seperate_kernels(self,k):
		return k[0:,0:,0:,0:,0], k[0:,0:,0:,0:,1]

	'''KL Conv Functions'''
	def cross_xl_kp_standalone(self,x):
		k0, k1 = self.seperate_kernels(self.get_prob_kernel())
		xp1 = x
		xp0 = 1 - x
		xp1.clamp_(epsilon, 1)
		xp0.clamp_(epsilon, 1)
		xlog1 = xp1.log()
		xlog0 = xp0.log()

		y0 = self.convwrap(xlog0, k0)
		y1 = self.convwrap(xlog1, k1)
		y = y0 + y1
		return y,[]
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
	def cross_xp_kl(self,xp1):
		xp0 = 1-xp1
		ent = - xp0*(xp0.clamp(epsilon,None).log()) - xp1*(xp1.clamp(epsilon,None).log()) #type:Tensor
		ent = ent.sum(dim=1,keepdim=True)
		lk = self.get_log_kernel()
		lk0 , lk1  = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0,lk0)
		y1 = self.convwrap(xp1,lk1)
		h = self.convwrap(ent, lk0[0:1, 0:1, 0:, 0:] * 0 + 1)
		h = h.sum(dim=1,keepdim=False).sum(dim=1,keepdim=False).sum(dim=1,keepdim=False)
		return y0 + y1,h
	def kl_xl_kp(self,x:torch.Tensor):
		'''conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor'''
		lk =self.get_log_kernel()
		pk = lk.exp()
		pk0,pk1 = self.seperate_kernels(pk)
		y = self.cross_xl_kp(x,pk0,pk1)
		y,ent = self.add_ker_ent(y,x,pk,lk)
		return y,ent

	def kl_xp_kl(self,xp1):
		xp0 = 1-xp1
		ent = - xp0*(xp0.clamp(epsilon,None).log()) - xp1*(xp1.clamp(epsilon,None).log()) #type:Tensor
		ent = ent.sum(dim=1,keepdim=True)
		lk = self.get_log_kernel()
		lk0 , lk1  = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0,lk0)
		y1 = self.convwrap(xp1,lk1)
		h = self.convwrap(ent,lk0[0:1,0:1,0:,0:]*0 + 1)
		y = y0+ y1 + h
		h = h.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return  y,h

	def kl_xp_kl_stoch(self, xp1):
		xp0 = 1 - xp1
		ent = - xp0 * (xp0.clamp(epsilon, None).log()) - xp1 * (xp1.clamp(epsilon, None).log())
		lk = self.get_log_kernel()
		lk0, lk1 = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0, lk0)
		y1 = self.convwrap(xp1, lk1)
		h = self.convwrap(ent, lk0[0:1, 0:1, 0:, 0:] * 0 + 1)
		return y0 + y1 + h

	def forward(self, x:torch.Tensor):
		if self.isrelu:
			y,ent = self.kl_xl_kp(x)
		else:
			y,ent = self.kl_xp_kl(x)

			# self.logprob = ent
		y = y + self.get_log_bias()

		return y



class JConv(KLConv_Base):
	''' Implements the convolutional probabilistic matching of input-filter
		JConv calculates the probability that a single sample generated from the filter matches
		a single sample generated from the input distribution and outputs the log probabiltiy.
	'''
	def __init__(self,
	             *args,
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
		super(JConv,self).__init__(**kwargs)
		self.paraminit = paraminit
		self.coefinit = 1
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.build()
	def crop_out(self,x):
		ksu = math.floor((self.kernel_shape[2]-1)/2)
		ksd = math.ceil((self.kernel_shape[2]-1)/2)
		return x[0:,0:,ksd:-ksu,ksd:-ksu]
	def forward(self, x:torch.Tensor):
		y = jointConv.apply(x,self.get_log_kernel())
		if self.padding[0]==0:
			y = self.crop_out(y)
		#y = self.convwrap(x.exp(),self.get_log_kernel().exp())
		return  y.log()# +self.get_log_bias()


class SpConv(KLConv_Base):
	''' When the Nonlinearity is the L2-Normalization, SPConv calculates the cosine similarity.'''
	def __init__(self,
	             *args,
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

		super(SpConv,self).__init__(**kwargs)
		self.paraminit = NormalParameter()
		self.isbiased = True
		self.build()

	def build(self):
		self.paraminit.coef = self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape,1))
		self.register_parameter('kernel', self.kernel)
		if self.isbiased:
			self.bias = Parameter(data=self.paraminit((1,self.kernel_shape[0],1,1),0.0001))
			self.register_parameter('bias', self.bias)

	def forward(self, x:torch.Tensor):
		kernel = self.paraminit.get_kernel(self.kernel)
		y = self.convwrap(x,kernel)
		y = y  + self.paraminit.get_kernel(self.bias)
		return y


class L2Norm(MyModule):
	def __init__(self, **kwargs):
		super(L2Norm, self).__init__(**kwargs)
	def forward(self, x:torch.Tensor):
		y = x / ((x**2).sum(dim=1, keepdim=True)).sqrt()
		return y


class Log(MyModule):
	def __init__(self, **kwargs):
		super(Log, self).__init__(**kwargs)
	def forward(self, x:torch.Tensor):
		y = x.clamp(epsilon,None).log()
		return y


class L2LogProb(MyModule):
	def __init__(self, **kwargs):
		super(L2LogProb, self).__init__(**kwargs)
	def forward(self, x:torch.Tensor):
		y = x**2 / ((x**2).sum(dim=1, keepdim=True)) #type: Tensor
		y = y.clamp(epsilon,None)
		y = y.log()
		return y


class KLAvgPool(MyModule):
	def __init__(self,spsize,stride,pad,isstoch=True, **kwargs):
		super(KLAvgPool,self).__init__(**kwargs)
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


class KLAvgPoolGL(MyModule):
	def __init__(self,*args,isstoch=False,**kwargs):
		super(KLAvgPoolGL,self).__init__(*args,**kwargs)
		self.isstoch = isstoch

	def forward(self, x:Tensor):
		einput = x.exp() #type: Tensor

		out = einput.mean(dim=2,keepdim=True)
		out = out.mean(dim=3, keepdim=True)
		out = out.clamp(epsilon,None)
		out = out.log()
		return out


class LNorm(MyModule):
	def __init__(self,isstoch,*args,**kwargs):
		super(LNorm,self).__init__(*args,**kwargs)
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


class Inp2Log(MyModule):
	def __init__(self):
		super(Inp2Log,self).__init__()
	def forward(self, x:Tensor):
		x= x.clamp(epsilon,None) # type: Tensor
		x = x/x.sum(dim=1,keepdim=True)
		out = x.log()

		return out


class Mixer(KLConv_Base):
	def __init__(self,
	             *args,
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
		super(Mixer,self).__init__(*args,**kwargs)
		paraminit.coef = 10
		self.paraminit = paraminit
		self.paraminit.isbinary = False # DO NOT Move these lines after super
		self.axisdim= 1
		self.use_conv = True

		if not self.use_conv:
			self.kernel = Parameter(data=self.paraminit((1,self.kernel_shape[1],1,1,self.kernel.size()[0])))
			self.register_parameter('kernel', self.kernel)
		else:
			self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
			self.register_parameter('kernel', self.kernel)
		if self.isbiased:
			self.paraminit.coef = 0
			self.biasinit = self.paraminit
			self.bias = Parameter(data=self.paraminit((1,self.kernel_shape[0],1,1)))
			self.register_parameter('bias', self.bias)


	def mix(self,x,lk:Tensor):
		ker_sz = lk.size()
		if ker_sz[2]*ker_sz[3] !=1:
			raise(Exception('Kernel size is not 1'))

		if  not  self.use_conv:
			x = x.unsqueeze(dim=4)
			if self.isstoch:
				y = LogSumExpStoch.apply(x+lk,1,1) #type:Tensor
				y = y.transpose(1,4)
				y = y.squeeze(dim=4)
			else:
				y = LogSumExp.apply(x+lk,1) # type: Tensor
				y = y.transpose(1, 4)
				y = y.squeeze(dim=4)
			return y
		else:
			y = mix.apply(x,self.get_log_kernel())
			#y = y + self.get_log_bias()
			#y = mix.normal_mix(x, self.get_log_kernel())
			#kerp = self.get_log_kernel().exp()
			#m = (x.max(dim=1,keepdim=True))[0]
			#x = x - m
			#x = x.exp()
			#y = F.conv2d(x,kerp)
			#y = (y ).log() + m
		return y


	def forward(self, x:torch.Tensor):
		y = self.mix(x,self.get_log_kernel())
		if self.isbiased:
			y = y + self.get_log_bias()
		return y

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
