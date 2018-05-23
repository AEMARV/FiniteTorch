from definition import *
from optstructs import *
import torch
import torch.nn
def quick_cifar():
	model_string = ''
	nl = 'relu'
	d = '->'
	finish = 'fin'
	model_string += 'conv|r:5,f:32,pad:same,bias:1' + d
	model_string += 'maxpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'conv|r:5,f:64,pad:same,bias:1' + d + nl
	model_string += 'avgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'conv|r:4,f:64,pad:same,bias:1' + d + nl
	model_string += 'conv|r:1,f:10,pad:same,bias:1' + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optimizer=None
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=150,
	                           batchsz=100,
	                           shuffledata=True,
	                           numworkers=1)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,optimizeropts=opts_optimizer,epocheropts=opts_epocher)
	return opts
