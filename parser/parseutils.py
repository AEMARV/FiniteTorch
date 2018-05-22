import torch
import torch.nn as nn
import torchvision as tv

def parse_model_string(modelstring,opts):
	layer_list_string = modelstring.split('->')
	layer_list = []
	in_n_channel = opts['inputspecs','in_n_channel']
	for blocknum,layer_string in enumerate(layer_list_string,0):
		layer_opts = parse_layer_opts(layer_string)
		if layer_string is 'fin':
			return layer_list
		elif layer_string is 'conv':
			ksize = layer_opts['r'].int()
			fnum = layer_opts['f'].int()
			layer += nn.Conv2d(in_n_channel,layer_opts['f'].int(),
			                   layer_opts)
		elif layer_string is 'relu':
		elif layer_string is 'lsoft':
		elif layer_string is 'maxpool':
		elif layer_string is 'avgpool':
		elif layer_string is 'margpool':
		return layer_list

def parse_layer_opts(layer_string):
	temp = layer_string.split('|')
	layer_opts_string = temp[1]
	layer_opts_list = layer_opts_string.split(',')
	layer_opts = {}
	for param_value in layer_opts_list:
		param_value_list = param_value.split(':')
		layer_opts[param_value_list[0]] = param_value_list[1]
	return layer_opts