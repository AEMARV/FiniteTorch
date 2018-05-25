import torch.nn as nn
from optstructs import *
from netparsers.parseutils import *
import torch.tensor
import math
class StaticNet(nn.Module):
	''' A static Module constructed from a model string. The model string specs are parsed with the static functions
	in the class.
	Constructor:

	StaticNet(modelstring, opts)

	modelstring: the model specification string with delimiter '->'
	opts: opts struct.
	'''
	def __init__(self,opts:NetOpts):
		super(StaticNet, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.opts = opts
		self.layerlist = self.parse_model_string(opts.modelstring,self.opts)
		for bloacknum,layer in enumerate(self.layerlist):
			self.add_module('block'+str(bloacknum),layer)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		for layer in self.layerlist:
			x = layer(x)
		return x

	''' String Parsers'''
	def parse_model_string(self, modelstring:str, opts:NetOpts):
		layer_list_string = modelstring.split('->')
		layer_list = []
		in_n_channel = opts.inputchannels
		out_n_channel = in_n_channel
		for blocknum, layer_string in enumerate(layer_list_string, 0):
			layer,out_n_channel = parse_layer_string(layer_string,out_n_channel)
			if layer is not None:
				layer_list += [layer]
		return layer_list
