import torch.nn as nn
from optstructs import *
from parser.parseutils import *
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

	def forward(self, x):
		# Max pooling over a (2, 2) window
		for layer in self.layerlist:
			x = layer(x)
		return x

	''' String Parsers'''
	@staticmethod
	def parse_model_string(modelstring:str, opts:NetOpts):
		layer_list_string = modelstring.split('->')
		layer_list = []
		in_n_channel = opts.inputchannels
		out_n_channel = in_n_channel
		for blocknum, layer_string in enumerate(layer_list_string, 0):
			layer_opts = parse_layer_opts(layer_string)
			if layer_string is 'fin':
				return layer_list,out_n_channel
			elif layer_string is 'conv':
				ksize = layer_opts['r'].int()
				fnum = layer_opts['f'].int()
				stride = layer_opts['stride'].int()
				pad = layer_opts['pad']
				pad = StaticNet.evalpad(pad)
				layer_list += nn.Conv2d(in_channels=in_n_channel, out_channels=fnum,
				                        kernel_size=ksize,
				                        stride=stride,
				                        padding=pad,
				                        bias=True)
			elif layer_string is 'relu':
				layer_list += nn.ReLU()
			elif layer_string is 'maxpool':
				ksize = layer_opts['r'].int()
				stride = layer_opts['stride'].int()
				pad = layer_opts['pad']
				pad = StaticNet.evalpad(pad)
				layer_list += nn.MaxPool2d(kernel_size=ksize,stride=stride,padding=pad)
			elif layer_string is 'avgpool':
				ksize = layer_opts['r'].int()
				stride = layer_opts['stride'].int()
				pad = layer_opts['pad']
				pad = StaticNet.evalpad(pad)
				layer_list += nn.AvgPool2d(kernel_size=ksize,stride=stride,padding=pad)
			else:
				raise(Exception('Undefin ed Layer: ' + layer_string))
