from definition import *


def quick_cifar(opts):
	model_string = ''
	nl = 'relu'
	d = '->'
	model_string = model_string + 'conv|r:5,f:32,pad:same,bias:1' + d
	model_string = model_string + 'maxpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string = model_string + 'conv|r:5,f:64,pad:same,bias:1' + d + nl
	model_string = model_string + 'avgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string = model_string + 'conv|r:4,f:64,pad:same,bias:1' + d + nl
	model_string = model_string + 'conv|r:1,f:10,pad:same,bias:1' + d

	''' Options'''
	opts[OPTS_MODEL] = model_string

	return model_string, opts
