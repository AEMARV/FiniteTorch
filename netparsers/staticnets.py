import torch.nn as nn
from optstructs import *
from netparsers.parseutils import *
import torch.tensor
import math
import torch.nn.functional as F
class StaticNet(MyModule):
	''' A static Module constructed from a model string. The model string specs are parsed with the static functions
	in the class.
	Constructor:

	StaticNet(modelstring, opts)

	modelstring: the model specification string with delimiter '->'
	opts: opts struct.
	'''
	def __init__(self,modelstring,inputchannels,weightinit=None,biasinit=None,sample_data=None):
		super(StaticNet, self).__init__(blockidx=0)
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.layerlist = self.parse_model_string(modelstring,inputchannels)
		for bloacknum,layer in enumerate(self.layerlist):
			if isinstance(layer,nn.Conv2d):
				weightinit(layer.weight.data)
				biasinit(layer.bias.data)

			self.add_module('block'+str(bloacknum),layer)

	def generate(self,y):

		for l in reversed(self.layerlist):
			if isinstance(l, torch.nn.Conv2d):
				#y = y - l.add_bias(y,)
				bias_reshaped = l.bias.view(1,y.shape[1],1,1)
				y = y - bias_reshaped.data
				wnorm = (l.weight.data**2).sum(dim=1,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True).sqrt()
				w = l.weight.data/wnorm

				weightmat = l.weight.view(3,3).inverse().view(3,3,1,1)
				y = F.conv_transpose2d(y,weightmat)
			elif isinstance(l, torch.nn.Sigmoid):
				y=y

			else:
				y= l.generate(y)


		return y

	def forward(self, x:Tensor,MAP=False):
		# Max pooling over a (2, 2) window
		logprob = 0
		model_prob = 0
		for layer in self.layerlist:
			if not isinstance(layer,MyModule):
				x = layer(x)
				logprob_temp=0
				model_prob_tmp=0
			else:
				x,logprob_temp,model_prob_tmp = layer(x)
			if hasnan(x):
				raise Exception(str(layer) + 'has nan')
			logprob += logprob_temp
			model_prob += model_prob_tmp
		return x,logprob, model_prob

	''' String Parsers'''
	def parse_model_string(self, modelstring:str, in_n_channel):
		layer_list_string = modelstring.split('->')
		layer_list = []
		out_n_channel = in_n_channel
		blockidx_dict= {}
		in_icnum=3
		for blocknum, layer_string in enumerate(layer_list_string, 0):
			layer,out_n_channel,in_icnum= parse_layer_string(layer_string,out_n_channel,in_icnum,blockidx_dict)
			if layer is not None:
				layer_list += [layer]
		return layer_list
class DenseNet(StaticNet):
	def __init__(self,*args,sample_data=None,**kwargs):
		super(DenseNet, self).__init__(*args,**kwargs)
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		convdata = [sample_data]
		self.forward(sample_data.to(device=torch.device('cuda:0')))
		pass
	def forward(self, x:Tensor):
		# Max pooling over a (2, 2) window
		logprob = 0
		x,_ = sample(x,1,1)
		x = x.log()
		convdata=[]
		for layer in self.layerlist:
			if isinstance(layer, KLConv_Base):
				xnew , logprob_temp = layer([x] +convdata)
				xavgd,logprob_temp2 = sample(glavgpool.apply(x)[0],1,1)
				logprob += logprob_temp2
				convdata = [xavgd.log()] + convdata
				x = xnew
			else:
				x,logprob_temp = layer(x)
			logprob += logprob_temp
		return x,logprob
class DeepSoup(StaticNet):
	pass
