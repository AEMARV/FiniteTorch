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
from layers.pmaputils import sample
import torch.autograd.gradcheck
import os
from torchvision.utils import save_image,make_grid
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
				temp = m.get_reg_vals()
				if temp is not None:
					reg = reg + temp
		self.regularizer = 0
		return reg
	def print(self,inputs,epoch_num,batch_num):
		for m in self.children():
			inputs,_ = m.forward(inputs)

			if isinstance(m, MyModule):
				m.print_output(inputs, epoch_num, batch_num)
				m.print_filts(epoch_num,batch_num)
	def get_scalar_dict(self):

		for m in self.children():
			if isinstance(m, MyModule):
				self.scalar_dict.update(m.get_scalar_dict())

		return self.scalar_dict

	def print_filts(self,epoch,batch):
		try:
			probkernel = self.get_prob_kernel()
		except:
			return
		sh = probkernel.shape
		probbias = self.get_log_bias().exp().view(sh[0], 1, 1, 1)
		probkernel = probkernel * probbias
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Filters' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/filt_' + str(epoch)+'_'+str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)
	def print_output(self, y,epoch,batch):
		probkernel = y.exp()
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)
class KLConv_Base(MyModule):
	def __init__(self,
	             *args,
	             fnum=None,
	             icnum=1,
	             inp_icnum,
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
		self.icnum=icnum
		self.inp_icnum = inp_icnum
		self.fnum = fnum
		self.chansz = inp_chan_sz
		self.coefinit  = coefinit
		self.kernel_shape = (fnum*self.icnum,)+(inp_chan_sz*inp_icnum,)+(kersize,kersize)
		self.paraminit = paraminit# type:Parameterizer
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
		self.paraminit.coef =  self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		self.register_parameter('kernel', self.kernel)
		if self.isbiased:
			self.paraminit.coef= 0
			self.bias = Parameter(data=self.paraminit((1,self.fnum,1,1,self.icnum),isbias=True))
			self.register_parameter('bias', self.bias)

	'''Kernel/Bias Getters'''
	def get_log_kernel(self,index=0):
		k = self.kernel
		sp1 = k.shape[2]
		sp2 = k.shape[3]
		k = k.reshape((self.fnum*self.icnum,self.chansz,self.inp_icnum,k.shape[2],k.shape[3]))
		k = self.paraminit.get_log_kernel(k)
		return k
	def get_log_kernel_conv(self,):
		k = self.get_log_kernel()
		k = k.permute((0, 1, 2, 3, 4))
		k = k.reshape((self.fnum * self.icnum, self.inp_icnum * self.chansz, self.kernel.shape[2], self.kernel.shape[3]))
		return k

	def get_prob_kernel(self)-> torch.Tensor:

		return self.get_log_kernel().exp()

	def get_log_bias(self,index=0):
		return self.paraminit.get_log_kernel(self.bias)

	def get_prob_bias(self)-> torch.Tensor:
		return self.biasinit.get_prob_bias(self.bias)

	def convwrap(self,x:Tensor,w:Parameter):
		y = F.conv2d(x, w, bias=None,
		             stride=self.stride,
		             padding=self.padding)
		return y
	def reshape_input_for_conv(self,x:Tensor):
		if x.ndimension()<5:
			return x

		x=(x.permute(0,1,4,2,3))
		x = x.reshape((x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]))
		return x
	def reshape_input_for_nxt_layer(self,ly):
		ly = ly.view((ly.shape[0],self.fnum,self.icnum,ly.shape[2],ly.shape[3]))
		ly = ly.permute(0,1,3,4,2)
		return ly
	def add_ker_ent(self,y:torch.Tensor,x,pker,lker,mask=None):
		H = self.ent_per_spat(pker,lker)

		if mask is not None:
			H = self.convwrap((x[0:, 0:1, 0:, 0:]*0 + 1)*mask, H)
		else:
			H = self.convwrap(x[0:,0:1,0:,0:]*0 +1,H)
		hall = H.mean()
		return y + H,hall
	def project_params(self):
		self.kernel = self.paraminit.projectKernel(self.kernel)


	def get_log_prob(self):
		lprob = self.logprob
		self.logprob= 0
		return lprob
	'''Entorpy Functions'''
	def ent_per_spat(self,pker,lker):
		raise Exception(self.__class__.__name__ + ":Implement this function")
	def ent_kernel(self):
		return self.ent_per_spat(self.get_prob_kernel(),self.get_log_kernel()).sum()
	def expctd_ent_kernel(self):
		return self.ent_per_spat(self.get_log_kernel().exp(),self.get_log_kernel()).mean()/self.get_log_symbols()
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


class BayesFunc(KLConv_Base):
	def __init__(self,
	             *args,
	             paraminit=None,
	             islast=False,
	             write_images=False,
	             samplingtype=2,
	             exact=False,
	             **kwargs):
		'''      fnum=None,\n
	             kersize=None,\n
	             isbiased=False,\n
	             isrelu=True,\n
	             paraminit=None,\n
	             biasinit=None,\n
	             padding=None,\n
	             stride=1'''
		super(BayesFunc, self).__init__(*args, **kwargs)
		self.paraminit = paraminit
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim = 1
		self.build()
		self.exact = exact
		if exact and self.kernel_shape[2] >1:
			raise Exception("Exact Calculation of gradient is not possible with receptive filed >1")
		self.useless_counter = 0
		self.write_images = write_images
		self.samplingtype = samplingtype  # Mile Sampling is 1, Rejection Sampling is 0#
		if write_images:
			self.register_backward_hook(BayesFunc.print_grad_out)
			self.register_backward_hook(BayesFunc.print_grad_filt)
		self.islast = islast

	def build(self):
		self.paraminit.coef = self.coefinit
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		# self.kernel = Parameter(data=torch.normal(torch.zeros(self.kernel_shape),self.coefinit))
		self.kernel.requires_grad = True
		self.register_parameter('kernel', self.kernel)
		self.paraminit.coef = 0
		self.bias = Parameter(data=self.paraminit((1, self.kernel_shape[0], 1, 1,self.icnum), isbias=True))
		self.bias.requires_grad = self.isbiased
		self.register_parameter('bias', self.bias)

	def update_scalar_dict(self, self2, input, output):
		return

	def print_filts(self,epoch,batch):
		probkernel = self.get_prob_kernel()
		sh = probkernel.shape
		probbias = self.get_log_bias().exp().view(sh[0], 1, 1, 1)
		probkernel = probkernel * probbias
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Filters' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/filt_' + str(epoch)+'_'+str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)

	def print_output(self, y,epoch,batch):
		probkernel = y.exp()
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)

	@staticmethod
	def print_grad_out(self, grad_input, grad_output):
		grad_output = -grad_output[0]
		sh = grad_output.shape
		grad_output = grad_output.view(sh[0] * sh[1], 1, sh[2], sh[3])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'GradOutput' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/gradout_' + str(self.useless_counter) + '.bmp'

		save_image(grad_output, imagepath, normalize=True, scale_each=False, nrow=sh[1])

	@staticmethod
	def print_grad_filt(self, grad_input, grad_output):
		grad_output = -grad_input[1]
		sh = grad_output.shape
		grad_output = grad_output.view(sh[0] * sh[1], 1, sh[2], sh[3])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'GradFilt' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/gradfilt_' + str(self.useless_counter) + '.bmp'

		save_image(grad_output, imagepath, normalize=True, scale_each=False, nrow=sh[1])

	def testGrad(self, x):
		input = torch.rand(1, 2, 2, 2).to('cuda:0')
		input = (input / input.sum(dim=1, keepdim=True)).log()

		checks = torch.autograd.gradcheck(ConvBayesMap.apply,
		                                  [input, self.get_log_kernel(), self.get_log_bias(), 100, self.padding[0],
		                                   self.stride], eps=1e-2, atol=0.1, rtol=1e-1)
		if checks:
			print("Yes")
		else:
			print("Oh no")
		return checks

	#def get_log_kernel(self):
	#	norm = (self.kernel.abs()).mean()
	# print(norm.item())
	#	return self.kernel
	# def get_log_bias(self):
	#	return self.bias

	def calcent(self, y: Tensor):
		ent = -y.clamp(-10000, None) * y.exp()
		ent = ent.sum(dim=1, keepdim=True)
		ent = ent.mean()

		return ent

	def generate(self, y):
		x = ConvBayesMap.backward_func()

	def ent_filts(self, y):
		b = self.get_log_bias()
		self.get_log_kernel() + self.get_log_bias().view([])
		ent = -y.clamp(-10000, None) * y.exp()
		ent = ent.sum(dim=1, keepdim=True)
		ent = ent.mean()


	def get_log_kernel(self,index=0):
		k = self.kernel
		sp1 = k.shape[2]
		sp2 = k.shape[3]
		k = k.reshape((self.fnum*self.icnum,self.chansz,self.inp_icnum,k.shape[2],k.shape[3]))
		k = self.paraminit.get_log_kernel(k)
		return k
	def get_log_kernel_conv(self,):
		k = self.get_log_kernel()
		k = k.permute((0, 2, 1, 3, 4))
		k = k.reshape((self.fnum * self.icnum, self.inp_icnum * self.chansz, self.kernel.shape[2], self.kernel.shape[3]))
		return k

#def sample_log_likelihood(self, x):
#	pass
	def get_kernel_expanded_format(self,k):
		sp1 = k.shape[2]
		sp2 = k.shape[3]
		k = k.reshape((self.fnum * self.icnum, self.chansz, self.inp_icnum, k.shape[2], k.shape[3]))
		return k
	def reshape_input_for_conv(self,x:Tensor):
		if x.ndimension()<5:
			return x

		x=(x.permute(0,1,4,2,3))
		x = x.reshape((x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]))
		return x
	def reshape_input_for_nxt_layer(self,ly):
		ly = ly.view((ly.shape[0],self.fnum,self.icnum,ly.shape[2],ly.shape[3]))
		ly = ly.permute(0,1,3,4,2)
		return ly
	def get_reg_vals(self):
		k = self.get_kernel_expanded_format(self.kernel)
		k = self.paraminit.get_log_kernel(k)
		regs = (k).min(dim=1,keepdim=True)[0]
		return 0#self.regularizer
	def marvasti_divg(self):
		uniflogprob= float(-math.log(self.kernel.shape[1]))

		k = self.get_log_kernel()
		maxprob= (uniflogprob-k).min(dim=1,keepdim=True)[0]
		maxprob =maxprob.sum(dim=(2,3,4),keepdim=True)
		return maxprob
	def get_log_bias(self,index=0):
		k = self.paraminit.get_log_kernel(self.kernel)
		b = (-k).min(dim=1,keepdim=True)[0]
		b = b.sum(dim=(1,2,3),keepdim=True).permute((1,0,2,3))
		b = self.reshape_input_for_nxt_layer(b)
		b = b - b.logsumexp(dim=1,keepdim=True)
		return b
	def get_model_lprob(self):
		b = self.get_log_bias()
		k = self.get_log_kernel()
		blunif = -math.log(b.shape[1])
		klunif = -math.log(k.shape[1])

		bias_divg = blunif - b
		kernel_divg = klunif - k

		bias_divg = bias_divg.min(dim=1,keepdim=True)[0].sum()
		kernel_divg = kernel_divg.min(dim=1,keepdim=True)[0].sum()

		return bias_divg+kernel_divg


	def forward(self,lx,MAP=False):
		avglogprob = lx.mean(dim=1,keepdim=True)
		if not MAP:
			xsamp,logprob = sample(lx,1,1)
		else:
			xsamp, logprob = sample_map(lx, 1, 1)
		xsamp = self.reshape_input_for_conv(xsamp)
		if logprob.ndimension() ==4:
			logprob = logprob.sum(dim=(1,2,3),keepdim=True)
		else:
			logprob = logprob.sum(dim=(1, 2, 3,4), keepdim=True)
		ly = F.conv2d(xsamp,self.get_log_kernel_conv(),stride=self.stride,padding=self.padding[0])
		ly = self.reshape_input_for_nxt_layer(ly)
		##ly     +=  self.get_log_bias()# type:Tensor
		lnorm = ly.logsumexp(dim=1,keepdim=True)
		ly = ly - lnorm

		return ly,logprob.squeeze(), self.get_model_lprob()

	def forward2(self, x: torch.Tensor):
		# self.testGrad(x)
		log_prob = 0
		if self.exact:
			return self.forward_exact(x),0
		#self.useless_counter += 1
		if self.training:
			y = ConvBayesMap.apply(x, self.get_log_kernel(), self.get_log_bias(), 1, self.padding[0], self.stride,
			                       self.samplingtype)
		# self.regularizer = self.sample_log_prob()
		else:
			y = ConvBayesMap.apply(x, self.get_log_kernel(), self.get_log_bias(), 1, self.padding[0], self.stride,
			                       self.samplingtype)
		if self.write_images:
			self.print_filts()
			self.print_output(y)

		return y,0

	def forward_exact(self, x):
		bias = self.get_log_bias()
		kernel = self.get_log_kernel()
		bias = bias.view([kernel.shape[0], 1, 1, 1])
		kernel = kernel + bias
		kernel = kernel - LogSumExp.apply(kernel, 0, 1)
		y = F.conv2d(x.exp(), kernel.exp(), padding=self.padding).log()
		return y,0

class BayesFuncDense(BayesFunc):
	def __init__(self,*args,**kwargs):
		super(BayesFuncDense,self).__init__(*args,**kwargs)
		self.built=False
	def build(self):

		pass
	def buildkernels(self,tensorlist:List):
		kerspsz = self.kernel_shape[2]
		keroutsz = self.kernel_shape[0]
		self.paraminit.coef = self.coefinit
		if self.built:
			return
		self.kernel = []
		self.bias = []
		for i,tensor in enumerate(tensorlist):
			if i==1:
				kerspsz=1
			kernel = Parameter(data=self.paraminit((keroutsz,tensor.shape[1],kerspsz,kerspsz)))
			kernel.requires_grad=True
			self.register_parameter('kernel'+str(i),kernel)
			self.kernel = self.kernel + [kernel]

			bias = Parameter(data=self.paraminit((1, kernel.shape[0], 1, 1), isbias=True))
			bias.requires_grad = self.isbiased
			self.register_parameter('bias'+str(i), bias)
			self.bias =  self.bias+[bias]
		self.to('cuda:0')
		self.built=True
	def get_log_kernel(self,index=0):
		k =self.paraminit.get_log_kernel(self.kernel[index])
		return k
	def get_log_bias(self,index=0):
		b = self.paraminit.get_log_kernel(self.bias[index])
		return b
	def forward(self,lxlist:List):
		ly= 0
		logprob= 0
		self.buildkernels(lxlist)
		pad = self.padding[0]
		for i, lx in enumerate(lxlist):
			if i!=0:
				#lx,_ = glavgpool.apply(lx)
				pad =0
			#xsamp = lx.exp()
			xsamp, logprob_temp = sample(lx, 1, 1)
			#logprob += logprob_temp.sum(dim=(1, 2, 3), keepdim=True)
			convres = F.conv2d(xsamp, self.get_log_kernel(index=i), stride=self.stride,
			              padding=pad)
			ly = ly + convres
			#ly+=  self.get_log_bias(index=i)  # type:Tensor
		lnorm = ly.logsumexp(dim=1, keepdim=True)
		ly = ly - lnorm
		lysamp , logprob_temp = sample(ly,1,1)
		ly = lysamp.log()
		logprob += logprob_temp.sum(dim=(1, 2, 3), keepdim=True)
		return ly, logprob



class KLConv(KLConv_Base):
	def __init__(self,*args,
	             drop_rate=1,
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
		self.drop_rate = drop_rate
		self.conc_par= Parameter(data= torch.zeros(1))
		self.register_parameter('conc',self.conc_par)
		self.build()


	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''
	def kl_xl_kp_stoch(self,x:torch.Tensor):
		lkernel = self.get_log_kernel()
		sample_kernel,logprob = ExpStoch.apply(lkernel,1)
		self.logprob = logprob.sum()
		cross_y = self.convwrap(x, sample_kernel)
		kl,n = self.add_ker_ent(cross_y,x, sample_kernel, lkernel)
		return kl
	def kl_xl_kp(self,x:torch.Tensor,mask=1):
		''' Relu KL Conv '''
		lkernel = self.get_log_kernel_conv()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y, x, pkernel, lkernel,mask=mask)
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
		cross = self.convwrap(xp,self.get_log_kernel_conv())
		ent = self.convwrap(ent,self.get_log_kernel_conv().exp()[0:1,0:1,0:,0:]*0 + 1)
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
			xp,lprob = ExpStoch.apply(xl, 1)
			self.logprob = lprob.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
			ent = 0

		else:
			xp= xl.exp()
			ent = -xp * xl  # type: Tensor
			ent = ent.sum(dim=1, keepdim=True)
			ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)

		cross = self.convwrap(xp,self.get_log_kernel())

		return cross+ ent

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

	def get_mask(self,x):
		if self.drop_rate == 0:
			return 1
		else:
			if self.training:

				y = torch.rand_like(x[0:, 0:1, 0:, 0:,0:])
				y = (y < self.drop_rate).to(dtype=torch.float32)
				return y
			else:
				return 1
	def generate(self,y):
		y.detach_()
		#y = LogSumExpStoch.sample(y,1)
		y = y.exp()
		ysum = y.sum(dim=1,keepdim=True)
		y = (y+epsilon)/(ysum+epsilon)
		y = sampleprob(y,1,1)[0]
		lkernel = self.get_log_kernel()
		lkernel.detach_()
		x = F.conv_transpose2d(y,self.get_log_kernel(),stride=self.stride, padding=self.padding)
		#x=  (x-LogSumExp.apply(x,1,1)).exp()
		#x = x/(x+definition.epsilon).sum(dim=1,keepdim=True)
		return x
	def forward(self, x:torch.Tensor,MAP=False):
		#dummy = x[0:,0:1,0:,0:,0:]*0 +1
		#dummy = F.dropout(dummy,0.5,self.training)
		#x = x * dummy
		x= self.reshape_input_for_conv(x)
		#self.project_params()
		if not self.isrelu:
			# Sigmoid Activated

			y,g = self.kl_xp_kl(x)
				#self.logprob = self.ent_kernel()
		else:
			# ReLU Activated

			y,ent = self.kl_xl_kp(x)

		y = self.reshape_input_for_nxt_layer(y)
		#y = y*math.log(self.fnum)
		#y = y *((self.conc_par.exp()).exp())
		if self.isbiased:
			pass
			y = y+  self.get_log_bias()

		#y = y + self.get_log_bias()
		return y,0,self.paraminit.get_log_prior(self.kernel)*0


class MDConv(KLConv_Base):
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
		super(MDConv,self).__init__(*args,**kwargs)
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.build()
	def crop_out(self,x):
		if self.padding[0] !=0 :
			return x

		ksu = math.floor((self.kernel_shape[2]-1)/2)
		ksd = math.ceil((self.kernel_shape[2]-1)/2)
		return x[0:,0:,ksd:-ksu,ksd:-ksu]

	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''
	def kl_xl_kp_stoch(self,x:torch.Tensor):
		lkernel = self.get_log_kernel()
		sample_kernel,logprob = ExpStoch.apply(lkernel,1)
		self.logprob = logprob.sum()
		cross_y = self.convwrap(x, sample_kernel)
		kl,n = self.add_ker_ent(cross_y,x, sample_kernel, lkernel)
		return kl
	def kl_xl_kp(self,x:torch.Tensor):
		''' Relu KL Conv '''
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y, x, pkernel, lkernel)
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
			xp,lprob = ExpStoch.apply(xl, 1)
			self.logprob = lprob.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
			ent = 0

		else:
			xp= xl.exp()
			ent = -xp * xl  # type: Tensor
			ent = ent.sum(dim=1, keepdim=True)
			ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)

		cross = self.convwrap(xp,self.get_log_kernel())

		return cross+ ent

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

	def get_reg_vals(self):
		return None
		#TODO: Calculate the md prob of the mixture model
		b = self.get_log_bias() #type: Tensor
		if type(b) is not float:
			b = b.permute((1,0,2,3))
		k = self.get_log_kernel()
		k = k+b # type: Tensor
		meank = LogSumExp.apply(k,0)  #type: Tensor
		return meank.max(dim=1,keepdim=True)[0].sum()
	def forward(self, x:torch.Tensor):
		#self.project_params()
		if not self.isrelu:
			# Sigmoid Activated
			y = md_conv.apply(x,self.get_log_kernel())
		else:
			# ReLU Activated
			y = mdi_conv.apply(x,self.get_log_kernel())
		y = self.crop_out(y)
		y = y + self.get_log_bias()

		return y


class Classifier(KLConv_Base):
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
		super(Classifier, self).__init__(*args,**kwargs)
		self.paraminit.isbinary = False  # DO NOT Move these lines after super
		self.axisdim= 1
		self.build()
	def crop_out(self,x):
		if self.padding[0] !=0 :
			return x

		ksu = math.floor((self.kernel_shape[2]-1)/2)
		ksd = math.ceil((self.kernel_shape[2]-1)/2)
		return x[0:,0:,ksd:-ksu,ksd:-ksu]

	''' Ent Functions'''
	def ent_per_spat(self,pker,lker):
		# Entropy Per spatial Position
		H = pker*lker
		H = -H.sum(dim=1,keepdim=True)
		return H

	'''Conv Functions'''
	def kl_xl_kp_stoch(self,x:torch.Tensor):
		lkernel = self.get_log_kernel()
		sample_kernel,logprob = ExpStoch.apply(lkernel,1)
		self.logprob = logprob.sum()
		cross_y = self.convwrap(x, sample_kernel)
		kl,n = self.add_ker_ent(cross_y,x, sample_kernel, lkernel)
		return kl

	def kl_xl_kp(self,x:torch.Tensor):
		''' Relu KL Conv '''
		lkernel = self.get_log_kernel()
		pkernel = lkernel.exp()
		y = self.convwrap(x,pkernel)
		y,ent = self.add_ker_ent(y, x, pkernel, lkernel)
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
			xp,lprob = ExpStoch.apply(xl, 1)
			self.logprob = lprob.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
			ent = 0

		else:
			xp= xl.exp()
			ent = -xp * xl  # type: Tensor
			ent = ent.sum(dim=1, keepdim=True)
			ent = self.convwrap(ent, self.get_prob_kernel()[0:1, 0:1, 0:, 0:] * 0 + 1)

		cross = self.convwrap(xp,self.get_log_kernel())

		return cross+ ent

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

	def get_reg_vals(self):
		return None

	def forward(self, x:torch.Tensor):
		xsampled = ExpStoch.apply(x,1)
		if not self.isrelu:
			# Sigmoid Activated
			y = md_conv.apply(x,self.get_log_kernel())
		else:
			# ReLU Activated
			y = mdi_conv.apply(x,self.get_log_kernel())
		y = self.crop_out(y)
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
		xp1 = xp1.clamp(epsilon,1)
		xp0 = xp0.clamp(epsilon,1)
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

	def kl_xl_kp_stoch(self,x:torch.Tensor):
		return None

	def kl_xp_kl(self,xp1):
		xp0 = 1-xp1
		ent = - xp0*(xp0.clamp(epsilon,None).log()) - xp1*(xp1.clamp(epsilon,None).log()) #type:Tensor
		ent = ent.sum(dim=1,keepdim=True)
		lk = self.get_log_kernel()
		lk0 , lk1 = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0,lk0)
		y1 = self.convwrap(xp1,lk1)
		h = self.convwrap(ent,lk0[0:1,0:1,0:,0:]*0 + 1)
		y = y0+ y1 + h
		h = h.sum(dim=1, keepdim=False).sum(dim=1, keepdim=False).sum(dim=1, keepdim=False)
		return y,h

	def kl_xp_kl_stoch(self, xp1):
		xp0 = 1 - xp1
		ent = - xp0 * (xp0.clamp(epsilon, None).log()) - xp1 * (xp1.clamp(epsilon, None).log())
		lk = self.get_log_kernel()
		lk0, lk1 = self.seperate_kernels(lk)
		y0 = self.convwrap(xp0, lk0)
		y1 = self.convwrap(xp1, lk1)
		h = self.convwrap(ent, lk0[0:1, 0:1, 0:, 0:] * 0 + 1)
		return y0 + y1 + h
	def generate(self,y):
		#y = LogSumExpStoch.sample(y, 1)
		#y= y.exp()
		#ysum = y.sum(dim=1, keepdim=True)
		#y = (y + epsilon) / (ysum + epsilon)
		#y = sampleprob(y, 1, 1)[0]
		#sampleker0,sampleker1 = self.seperate_kernels(sample(self.get_log_kernel(),4,50)[0])
		lker0,lker1 = self.seperate_kernels(self.get_log_kernel())
		x1 = F.conv_transpose2d(y,lker0.exp(),stride = self.stride, padding=self.padding)
		x0 = F.conv_transpose2d(y, lker1.exp(), stride=self.stride, padding = self.padding)
		x = x1 -x0
		#x = (x1/x0).log()
		#x0 = F.conv_transpose2d(y, lker0, stride=self.stride, output_padding=0)
		#x  = F.sigmoid(x-x0)
		return x
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


class NonFactMarkov(MyModule):
	def __init__(self,*args,fnum=-1,kersize=-1,inp_chan_sz=-1,**kwargs):
		super(NonFactMarkov,self).__init__(*args,**kwargs)
		if True:
			self.kernel = Parameter(data= (0*torch.rand(fnum,inp_chan_sz,kersize,kersize)+1).log())

		else:
			self.kernel = Parameter(data=(torch.rand(fnum, inp_chan_sz, kersize, kersize)).log()/math.log(self.kernel.shape[1]))
		self.register_parameter('kernel', self.kernel)
	def get_log_kernel(self):
		k = self.kernel * math.log(self.kernel.shape[1])
		return k - LogSumExp.apply(k,0,1)
	def get_prob_kernel(self):
		return self.get_log_kernel().exp()
	def forward(self, x):
		x = x.exp()
		ker = self.get_prob_kernel()
		y = F.conv2d(x,ker,padding=0)
		y = y.log()
		return y


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


'''Pixel Wise Layers'''

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
		y = x / ((x+epsilon).sum(dim=1, keepdim=True)) #type: Tensor
		y = y.clamp(epsilon,None)
		y = y.log()
		return y
	def generate(self,y):
		return y


#TODO CHANGED FOR PMAPS, REVERSE BACK TO KLMAPS
class KLAvgPool(MyModule):
	def __init__(self,spsize,stride,pad,isstoch=True, **kwargs):
		super(KLAvgPool,self).__init__(**kwargs)
		self.spsize= spsize
		self.stride = stride
		self.pad = num_pad_from_symb_pad(pad,spsize)
		self.isstoch = isstoch

	def forward(self, x:Tensor,MAP=False):
		log_prob= 0
		chans = x.shape[1]
		icnum = x.shape[4]
		#einput, log_prob = sample(x, 1, 1)
		einput = x.exp()
		einput = einput.permute(0,1,4,2,3)
		einput = einput.reshape((einput.shape[0],einput.shape[1]*einput.shape[2],einput.shape[3],einput.shape[4]))

		out = F.avg_pool2d(einput,
		                   self.spsize,
		                   stride=self.stride,
		                   padding=self.pad,
		                   count_include_pad=False)
		out = out.reshape((out.shape[0],chans,icnum,out.shape[2],out.shape[3]))
		out = out.permute(0,1,3,4,2)
		out = out.clamp(epsilon,None)
		out = out.log()
		return out,log_prob,0#.sum(dim=(1,2,3,4),keepdim=True).squeeze()
	def generate(self,y:Tensor):
		#y = LogSumExpStoch.sample(y,1)
		y = y.exp()
		x = F.upsample(y,scale_factor=self.stride,mode='bilinear')
		x = x.log()
		return x
	def print_filts(self,epoch,batch):
		pass

	def print_output(self, y,epoch,batch):
		probkernel = y.exp()
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)

#TODO CHANGED FOR PMAPS, REVERSE BACK TO KLMAPS
class KLAvgPoolGL(MyModule):
	def __init__(self,*args,isstoch=False,**kwargs):
		super(KLAvgPoolGL,self).__init__(*args,**kwargs)
		self.isstoch = isstoch

	def forward(self, x:Tensor,MAP=False):
		log_prob = 0
		#einput,log_prob = sample(x,1,1) #type: Tensor
		einput = x.exp()
		out = einput.mean(dim=(2,3,4),keepdim=True)
		#out = out.mean(dim=3, keepdim=True)
		#out = out.clamp(epsilon,None)
		out = out.log()

		return out,log_prob,0
	def print_filts(self,epoch,batch):
		pass

	def print_output(self, y,epoch,batch):
		probkernel = y.exp()
		chans = probkernel.shape[1]
		probkernel = probkernel.view(
			[probkernel.shape[0] * probkernel.shape[1], 1, probkernel.shape[2], probkernel.shape[3]])
		dirpath = './GenImages/' + self.compact_name + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		dirpath = dirpath + 'Output' + '/'
		if not os.path.exists(dirpath):
			os.mkdir(dirpath)
		imagepath = dirpath + '/output_' + str(epoch) + '_' + str(batch) + '.bmp'

		save_image(probkernel, imagepath, normalize=True, scale_each=False, nrow=chans)

class LNorm(MyModule):
	def __init__(self,isstoch,*args,isregulated=False,**kwargs):
		super(LNorm,self).__init__(*args,**kwargs)
		self.isstoch= isstoch
		self.isregulated= isregulated
		if not(self.isstoch):
			self.implicit_layer = nn.LogSoftmax(dim=1)
			self.add_module('logsoft',self.implicit_layer)
	def forward(self, x,MAP=False):
		if self.isstoch:
			m = LogSumExpStoch.apply(x, 1,1)
			out = x - m
		else:
			#out = self.implicit_layer(x)
			out = x - x.logsumexp(dim=1,keepdim=True)
			if definition.hasnan(out):
				print("DUDE")
				pass

		return out,0,0
	def generate(self,y):
		y = (y - LogSumExp.apply(y,1,1))
		return y


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



class Transitioner(KLConv_Base):
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
		super(Transitioner,self).__init__(*args,**kwargs)
		self.paraminit = paraminit
		self.paraminit.isbinary = False # DO NOT Move these lines after super
		self.axisdim= 1
		self.use_conv = True
		self.kernel = Parameter(data=self.paraminit(self.kernel_shape))
		self.register_parameter('kernel', self.kernel)
		self.build()
	def update_scalar_dict(self, self2, input, output):
		return

	def forward(self, x:torch.Tensor):
		lkernel = self.kernel
		lkernel = lkernel - LogSumExp.apply(lkernel,0)
		y= self.convwrap(x.exp(),lkernel.exp()).log()
		return y


class JSDivg(Module):
	def __init__(self,*args,**kwargs):
		super(JSDivg,self).__init__()
	def forward(self, x,target):
		target = target.unsqueeze(1)
		target_one_hot = x.new_zeros([x.size()[0], x.size()[1]])  # type: Tensor
		target_one_hot.scatter_(1, target, 1)
		probs = x.exp()
		mixed = (probs + target_one_hot) / 2
		divg1 = (mixed.log() - x)  # type: Tensor
		divg1, d = divg1.min(dim=1, keepdim=True)
		divg2 = (mixed.log() - target_one_hot.log())  # type: Tensor
		divg2, d = divg2.min(dim=1, keepdim=True)
		return -(divg1 + divg2)



def num_pad_from_symb_pad(pad:str,ksize:int)->Tuple[int]:
	if pad=='same':
		rsize = ksize
		csize = ksize
		padr = (rsize-1)/2
		padc = (csize-1)/2
		return (int(padr),int(padc))
	elif pad=='valid':
		padr=0
		padc=0
		return (int(padr),int(padc))
	elif type(pad) is tuple:
		return pad
	else:
		raise(Exception('Padding is unknown--Pad:',pad))
