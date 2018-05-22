import torch.nn as nn


class StaticNet(nn.Module):
	''' A static Module constructed from a model string. The model string specs are parsed with the static functions
	in the class.
	Constructor:

	StaticNet(modelstring, opts)

	modelstring: the model specification string with delimiter '->'
	opts: opts struct.
	'''
	def __init__(self,modelstring,opts):
		super(StaticNet, self).__init__()
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.opts = opts
		self.modelstring = modelstring
		self.layerlist = self.parse_model_string(modelstring)

	def forward(self, x):
		# Max pooling over a (2, 2) window
		for layer in self.layerlist:
			x = layer(x)
		return x

	''' String Parsers'''
	@staticmethod
	def parse_model_string(modelstring, opts):
		layer_list_string = modelstring.split('->')
		layer_list = []
		in_n_channel = opts['inputspecs', 'in_n_channel']
		out_n_channel = in_n_channel
		for blocknum, layer_string in enumerate(layer_list_string, 0):
			layer_opts = StaticNet.parse_layer_opts(layer_string)
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
			elif layer_string is 'maxpool':
				ksize = layer_opts['r'].int()
				stride = layer_opts['stride'].int()
				pad = layer_opts['pad']
				pad = StaticNet.evalpad(pad)
				layer_list += nn.AvgPool2d(kernel_size=ksize,stride=stride,padding=pad)
			else:
				raise('Undefined Layer: ' + layer_string)

	@staticmethod
	def parse_layer_opts(layer_string):
		temp = layer_string.split('|')
		layer_opts_string = temp[1]
		layer_opts_list = layer_opts_string.split(',')
		layer_opts = {}
		for param_value in layer_opts_list:
			param_value_list = param_value.split(':')
			layer_opts[param_value_list[0]] = param_value_list[1]
		return layer_opts

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features