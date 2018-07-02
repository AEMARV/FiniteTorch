from optstructs import *
import torch
import torch.nn

''' Available Options for KLCONV('klconv'):
	r: receptive field,
	f: number of filters,
	pad: padding amount either 'same' or 'valid'
	bias: does it have bias (Bias not implemented yet)

	param: selects the parameterization methods including->

	 1. 'logstochu': log parameterization, stochastic gradient
	 and all dists are initialized to uniform
	 2. 'logstoch' : log parameterizaiton, stochastic gradient
	 3. 'log' : log parametrizaiton, exponential dists initialization  



	'''
def hello_stoch_hello_kl() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'lnorm|s:0'
	convparams = 'param:logunif,stoch:0'
	finish = 'fin'
	model_string += 'inplog'+d
	model_string += 'klconv|r:3,f:32,pad:same,bias:0,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1,stoch:1'    + d + nl + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1,stoch:1'         + d + nl + d
	model_string += 'klconv|r:1,f:10,pad:valid,bias:0,{}'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:4,pad:valid,stride:1,bias:1,stoch:1,isglobal:true' + d + nl + d
	model_string += finish

	''' Net Options'''
	opts_net = NetOpts(model_string,
	                   inputchannels=3,
	                   inputspatsz=32)
	'''Optimizer Options'''
	opts_optim =OptimOpts(lr=1,
	                      type='SGD',
	                      momentum=0.9,
	                      weight_decay=0,
	                      dampening=0,
	                      nestrov=False)
	'''Epocher Options'''
	opts_epocher = EpocherOpts(epochnum=3000,
	                           batchsz=100,
	                           shuffledata=True,
	                           loss=torch.nn.NLLLoss(),
	                           numworkers=1,
	                           gpu=True)
	''' Create All opts'''
	opts = allOpts(netopts=opts_net,
	               optimizeropts=opts_optim,
	               epocheropts=opts_epocher,
	               )
	return opts
