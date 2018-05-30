'''All the parser functions is implemented here'''
import torch.nn as nn
import math
from layers.klmodules import *
from layers.Initializers import *

def parse_layer_opts(layer_string):
	'''input is a layer description string with name|p1:[v1],p2[v2]... convention'''
	layer_string.rstrip(' ')
	temp = layer_string.split('|')
	layer_name_str = temp[0]
	if len(temp)>1:
		layer_opts_string = temp[1]
		layer_opts_list = layer_opts_string.split(',')
	else:
		layer_opts_list =[]
	layer_opts = {}
	for param_value in layer_opts_list:
		param_value_list = param_value.split(':')
		layer_opts[param_value_list[0]] = param_value_list[1]
	return layer_name_str,layer_opts

def evalpad(pad, ksize):
	if pad == 'same':
		totalpad = ksize - 1
		padding = int(math.floor(totalpad / 2))
	else:
		padding = 0
	return padding

def get_init(initstring:str)->Parameterizer:
	if initstring=='logstochu':
		init = LogParameter(isstoch=True,isuniform=True)
	elif initstring == 'logstoch':
		init = LogParameter(isstoch=True, isuniform=False)
	elif initstring == 'log':
		init = LogParameter(isstoch=False, isuniform=False)
	else:
		raise(Exception('Unknown Parameterizer: '+initstring))
	return init

def parse_layer_string(layer_string,in_n_channel):
	out_n_channel = -1
	layer_name_str,layer_opts = parse_layer_opts(layer_string)
	if layer_name_str == 'fin':
		return None,in_n_channel
	# -------------------------------------------------------------------Finite Convs
	elif layer_name_str == 'klconv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		param = get_init(layer_opts['param'])
		layer = KLConv(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               isbiased=False,
		               isrelu=True,
		               biasinit=None,
		               padding=pad,
		               paraminit=param)
		out_n_channel = fnum
	elif layer_name_str == 'klconvb':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		param = get_init(layer_opts['param'])
		layer = KLConvB(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               isbiased=False,
		               isrelu=True,
		               biasinit=None,
		               padding=pad,
		               paraminit=param)
		out_n_channel = fnum
	# -------------------------------------------------------------------Finite POOLS
	elif layer_name_str == 'klavgpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		layer = KLAvgPool(spsize=ksize,stride=stride,pad=pad)
		out_n_channel = in_n_channel
	# -------------------------------------------------------------------Finite Activations
	elif layer_name_str == 'klavgpool':
		isstoch = bool(layer_opts['s'])
		layer = LNorm(isstoch=isstoch)
		out_n_channel = in_n_channel
		nn.NLLLoss
	# -------------------------------------------------------------------Conv Equipment

	elif layer_name_str == 'conv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.Conv2d(in_channels=in_n_channel,
						  out_channels=fnum,
						  kernel_size=ksize,
						  stride=stride,
						  padding=pad,
						  bias=True)
		out_n_channel = fnum
	elif layer_name_str == 'relu':
		layer = nn.ReLU()
		out_n_channel = in_n_channel
	elif layer_name_str == 'maxpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.MaxPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel
	elif layer_name_str == 'avgpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'])
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.AvgPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel
	else:
		raise(Exception('Undefined Layer: ' + layer_name_str))
	if out_n_channel == -1:
		raise('Output Channel Num not assigned in :' + layer_name_str)
	return layer, out_n_channel
