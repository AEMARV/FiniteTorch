'''All the parser functions is implemented here'''
import torch.nn as nn


def parse_layer_opts(layer_string):
	'''input is a layer description string with name|p1:[v1],p2[v2]... convention'''
	layer_string.rstrip(' ')
	temp = layer_string.split('|')
	layer_name_str = temp[0]
	layer_opts_string = temp[1]
	layer_opts_list = layer_opts_string.split(',')
	layer_opts = {}
	for param_value in layer_opts_list:
		param_value_list = param_value.split(':')
		layer_opts[param_value_list[0]] = param_value_list[1]
	return layer_name_str,layer_opts


def evalpad(pad):
	return pad


def parse_layer_string(layer_string,in_n_channel,opts):
	out_n_channel = -1
	layer_name_str,layer_opts = parse_layer_opts(layer_string)
	if layer_name_str is 'fin':
		return None,in_n_channel
	elif layer_name_str is 'conv':
		ksize = layer_opts['r'].int()
		fnum = layer_opts['f'].int()
		stride = layer_opts['stride'].int()
		pad = layer_opts['pad']
		pad = evalpad(pad)
		layer = nn.Conv2d(in_channels=in_n_channel,
						  out_channels=fnum,
						  kernel_size=ksize,
						  stride=stride,
						  padding=pad,
						  bias=True)
		out_n_channel = fnum
	elif layer_name_str is 'relu':
		layer = nn.ReLU()
		out_n_channel = in_n_channel
	elif layer_name_str is 'maxpool':
		ksize = layer_opts['r'].int()
		stride = layer_opts['stride'].int()
		pad = layer_opts['pad']
		pad = evalpad(pad)
		layer = nn.MaxPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel
	elif layer_name_str is 'avgpool':
		ksize = layer_opts['r'].int()
		stride = layer_opts['stride'].int()
		pad = layer_opts['pad']
		pad = evalpad(pad)
		layer = nn.AvgPool2d(kernel_size=ksize, stride=stride, padding=pad)
		out_n_channel = in_n_channel
	else:
		raise('Undefined Layer: ' + layer_name_str)
	if out_n_channel == -1:
		raise('Output Channel Num not assigned in :' + layer_name_str)
	return layer, out_n_channel
