from optstructs import *
import torch
import torch.nn

''' Available Options for KLCONV('klconv'):
	r: receptive field,
	f: number of filters,
	pad: padding amount either 'same' or 'valid'
	bias: does it have bias (Bias not implemented yet)

	param: selects the parameterization methods including->

	 1. 'logstochunif': log parameterization, stochastic gradient
	 and all dists are initialized to uniform
	 2. 'stoch' : stochastic gradient
	 3. 'log' : log parametrizaiton, exponential dists initialization
	 4. 'sphere' : sphercial parameterization
	 5. 'unif' : sets the parameters to uniform dist
	   



	'''
def hello_stoch_quick_cifar_v0() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'lnorm|s:0'
	convparams = 'param:logdirich,stoch:0,isrelu:1'
	finish = 'fin'
	model_string += 'spconv|r:5,f:32,pad:same,bias:0,{}'.format(convparams) + d
	model_string += 'l2lpnorm' + d
	#model_string += 'mixer|f:32,param:log,stoch:1' + d
	#model_string += nl + d
	model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:5,f:64,pad:same,bias:0,{}'.format(convparams) + d
	#model_string += 'mixer|f:64,param:log,stoch:1' + d
	model_string += nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d
	model_string += 'klconv|r:4,f:64,pad:same,bias:0,{}'.format(convparams)  + d
	model_string += nl + d
	#model_string += 'mixer|f:64,param:log,stoch:1' + d + nl +d
	model_string += 'klconv|r:7,f:10,pad:valid,bias:0,param:log,stoch:0,isrelu:1'  +d
	#model_string += 'mixer|f:10,param:log,stoch:1' + d
	model_string += nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1'  + d
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
	                           batchsz=500,
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
def hello_l2() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'l2norm'
	convparams = 'param:logdirich,stoch:0,isrelu:1'
	finish = 'fin'
	model_string += 'spconv|r:5,f:32,pad:same,bias:0,{}'.format(convparams) + d
	model_string += 'l2norm' + d
	#model_string += 'mixer|f:32,param:log,stoch:1' + d
	#model_string += nl + d
	model_string += 'avgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
	model_string += 'spconv|r:5,f:64,pad:same,bias:0,{}'.format(convparams) + d
	#model_string += 'mixer|f:64,param:log,stoch:1' + d
	model_string += nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d
	model_string += 'spconv|r:4,f:64,pad:same,bias:0,{}'.format(convparams)  + d
	#model_string += 'mixer|f:64,param:log,stoch:1' + d + nl +d
	model_string += nl + d
	model_string += 'spconv|r:7,f:10,pad:valid,bias:0,param:log,stoch:0,isrelu:1'  +d
	#model_string += 'mixer|f:10,param:log,stoch:1' + d
	model_string += 'l2lpnorm' + d
	model_string += 'glklavgpool|r:3,pad:same,stride:2,bias:1'  + d
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
def hello_stoch_hello_kl_hello_mixer() -> allOpts:
	model_string = ''
	d = '->'
	nl = 'lnorm|s:1'
	convparams = 'param:logdirich,isrelu:1'
	finish = 'fin'
	model_string += 'klconvb|r:3,f:32,pad:same,bias:0,param:logdirich,stoch:0,isrelu:1'  + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{},stoch:0'.format(convparams)  + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{},stoch:0'.format(convparams)  + d
	model_string += 'klconv|r:1,f:64,pad:same,bias:0,{},stoch:0'.format(convparams) + d + nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2'+ d
	# .......
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:0,{},stoch:0'.format(convparams) + d + nl + d
	#model_string += 'mixer|f:64,param:logdiric,stoch:1' + d
	#model_string +=   nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2'    + d
	# .......
	model_string += 'klconv|r:3,f:128,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:256,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	#model_string += 'mixer|f:128,param:logdirich,stoch:1' + d

	#model_string += nl + d
	model_string += 'klavgpool|r:3,pad:same,stride:2' + d
	# .......
	model_string += 'klconv|r:3,f:256,pad:same,bias:0,{},stoch:0'.format(convparams) + d + nl + d
	model_string += 'klconv|r:3,f:128,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:64,pad:same,bias:0,{},stoch:0'.format(convparams) + d
	model_string += 'klconv|r:3,f:32,pad:same,bias:0,{},stoch:0'.format(convparams) + d + nl + d
	#model_string += 'mixer|f:128,param:logdirich,stoch:1' + d
	#model_string += nl + d
	#model_string += 'klavgpool|r:3,pad:same,stride:1' + d + nl + d
	# .......

	model_string += 'klconv|r:3,f:10,pad:same,bias:0,{},stoch:0'.format(convparams)  + d + nl + d
	#model_string += 'mixer|f:10,param:logdirich,stoch:1, bias:1' + d
	model_string += 'glklavgpool' +d +nl + d



	#model_string += 'klavgpool|r:2,pad:valid,stride:1,bias:1,stoch:1,isglobal:true' + d + nl + d
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
