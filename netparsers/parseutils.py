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
		if len(param_value_list)<2:
			raise Exception(param_value_list[0] + 'is not initialized')
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
	if 'stoch' in initstring:
		isstoch = True
	else:
		isstoch = False
	if 'unif' in initstring:
		isuniform = True
	else:
		isuniform = False
	if 'dirich' in initstring:
		isdirichlet = True
	else:
		isdirichlet = False
	if 'log' in initstring:
		if 'proj' in initstring:
			init = LogParameterProjector(isstoch=isstoch, isuniform=isuniform, isdirichlet=isdirichlet)
		else:
			init = LogParameter(isstoch=isstoch,isuniform=isuniform,isdirichlet=isdirichlet)
	elif 'sphere' in initstring:
		init = SphereParameter(isstoch=isstoch, isuniform=isuniform, isdirichlet=isdirichlet)
	return init

def parse_layer_string(layer_string,in_n_channel,in_icnum,blockidx_dict):
	out_n_channel = -1
	out_icnum = in_icnum
	layer_name_str,layer_opts = parse_layer_opts(layer_string)
	if layer_name_str not in blockidx_dict.keys():
		blockidx_dict[layer_name_str] = 1
	blockidx = blockidx_dict[layer_name_str]
	if layer_name_str == 'fin':
		return None,in_n_channel,out_icnum

	# -------------------------------------------------------------------Finite Convs
	elif layer_name_str == 'klconv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		indpt_components = int(layer_opts['icnum']) if 'icnum' in layer_opts.keys() else 1
		stride = int(layer_opts['stride']) if 'stride' in layer_opts.keys() else 1
		coef = float(layer_opts['coef'])
		isbiased = ((layer_opts['bias']=='1')) if 'bias' in layer_opts.keys() else False
		isrelu = bool(int(layer_opts['isrelu']))
		drop_prob = float(layer_opts['droprate']) if 'droprate' in layer_opts.keys() else 0
		#stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = KLConv(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               inp_icnum= in_icnum,
		               icnum=indpt_components,
		               isbiased=isbiased,
		               isrelu=isrelu,
		               biasinit=None,
		               padding=pad,
		               paraminit=param,
		               isstoch=stoch,
		               coefinit=coef,
		               stride=stride,
		               drop_rate = drop_prob,
		               blockidx=blockidx)
		out_n_channel = fnum
		out_icnum= indpt_components
	elif layer_name_str == 'map':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		coef = float(layer_opts['coef'])
		isrelu = bool(int(layer_opts['isrelu'])) if 'isrelu' in layer_opts.keys() else False
		islast = bool(int(layer_opts['islast'])) if 'islast' in layer_opts.keys() else False
		sampopt = int(layer_opts['sampopt']) if 'sampopt' in layer_opts.keys() else 0
		indpt_components = int(layer_opts['icnum']) if 'icnum' in layer_opts.keys() else 1
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		exact = bool(int(layer_opts['exact'])) if 'exact' in layer_opts.keys() else False
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = BayesFunc(fnum=fnum,
		                icnum=indpt_components,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               inp_icnum = in_icnum,
		               isbiased=isbiased,
		               isrelu=isrelu,
		               biasinit=param,
		               samplingtype=sampopt,
		               padding=pad,
		               stride=stride,
		               paraminit=param,
		               isstoch=stoch,
		               coefinit=coef,
		               exact=exact,
					   islast=islast,
		               blockidx=blockidx)
		out_n_channel = fnum
		out_icnum = indpt_components

	elif layer_name_str == 'mapi':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		coef = float(layer_opts['coef'])
		isrelu = bool(int(layer_opts['isrelu'])) if 'isrelu' in layer_opts.keys() else False
		islast = bool(int(layer_opts['islast'])) if 'islast' in layer_opts.keys() else False
		sampopt = int(layer_opts['sampopt']) if 'sampopt' in layer_opts.keys() else 0
		indpt_components = int(layer_opts['icnum']) if 'icnum' in layer_opts.keys() else 1
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		exact = bool(int(layer_opts['exact'])) if 'exact' in layer_opts.keys() else False
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = BayesFuncI(fnum=fnum,
		                icnum=indpt_components,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               inp_icnum = in_icnum,
		               isbiased=isbiased,
		               isrelu=isrelu,
		               biasinit=param,
		               samplingtype=sampopt,
		               padding=pad,
		               stride=stride,
		               paraminit=param,
		               isstoch=stoch,
		               coefinit=coef,
		               exact=exact,
					   islast=islast,
		               blockidx=blockidx)
		out_n_channel = fnum
		out_icnum = indpt_components
	elif layer_name_str =='sample':
		layer = Sampler(blockidx=blockidx)
		out_n_channel = in_n_channel
		out_icnum = in_icnum
	elif layer_name_str =='psample':
		layer = PriorSampler(blockidx=blockidx)
		out_n_channel = in_n_channel +1
		out_icnum = in_icnum
	elif layer_name_str =='rsample':
		layer = RejectSampler(blockidx=blockidx)
		out_n_channel = in_n_channel
		out_icnum = in_icnum

	elif layer_name_str == 'mdconv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		coef = float(layer_opts['coef'])
		isrelu = bool(int(layer_opts['isrelu']))
		#stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = MDConv(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               isbiased=False,
		               isrelu=isrelu,
		               biasinit=None,
		               padding=pad,
		               paraminit=param,
		               isstoch=stoch,
		               coefinit=coef,
		               blockidx=blockidx)
		out_n_channel = fnum

	elif layer_name_str == 'jconv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		isrelu = False
		#stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = JConv(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               isbiased=False,
		               isrelu=isrelu,
		               biasinit=None,
		               padding=pad,
		               paraminit=param,
		               isstoch=stoch,
		              blockidx=blockidx)
		out_n_channel = fnum

	elif layer_name_str == 'spconv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		isrelu = bool(int(layer_opts['isrelu'])) if 'isrelu' in layer_opts else True
		#stride = int(layer_opts['stride'] if 'stride' in layer_opts else 1)
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = SpConv(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               isbiased=False,
		               isrelu=isrelu,
		               biasinit=None,
		               padding=pad,
		               paraminit=param,
		               isstoch=stoch,
		               stride=stride,
		               blockidx=blockidx)
		out_n_channel = fnum

	elif layer_name_str == 'klconvb':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		stride = int(layer_opts['stride'])
		coef = float(layer_opts['coef'])
		#stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		isbiased = ((layer_opts['bias'] == '1')) if 'bias' in layer_opts.keys() else False
		isrelu = bool(int(layer_opts['isrelu']))
		pad = layer_opts['pad']
		stoch = bool(layer_opts['stoch']=='1') if 'stoch' in layer_opts else False
		param = get_init(layer_opts['param'])
		layer = KLConvB(fnum=fnum,
		               kersize=ksize,
		               inp_chan_sz=in_n_channel,
		               isbiased=isbiased,
		               isrelu=isrelu,
		               biasinit=param,
		               padding=pad,
		               paraminit=param,
		                isstoch=stoch,
		                coefinit=coef,
		                stride=stride,
		                blockidx=blockidx)
		out_n_channel = fnum
	elif layer_name_str == 'mixer':
		ksize = 1
		fnum = int(layer_opts['f'])
		stoch = bool(layer_opts['stoch']=='1')
		bias = bool(layer_opts['stoch'] == '1')
		param = get_init(layer_opts['param'])
		layer = Mixer(fnum=fnum,
		              kersize=1,
		              inp_chan_sz=in_n_channel,
		              isbiased=bias,
		              paraminit=param,
		              isstoch=stoch,
		              blockidx=blockidx)
		out_n_channel = fnum
	elif layer_name_str == 'transit':
		ksize = 1
		fnum = int(layer_opts['f'])
		stoch = bool(layer_opts['stoch']=='1')
		bias = bool(layer_opts['stoch'] == '1')
		param = get_init(layer_opts['param'])
		layer = Transitioner(fnum=fnum,
		              kersize=1,
		              inp_chan_sz=in_n_channel,
		              isbiased=False,
		              paraminit=param,
		              isstoch=stoch,
		              blockidx=blockidx)
		out_n_channel = fnum

	elif layer_name_str == 'nfmarkov':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		layer = NonFactMarkov(fnum=fnum,
			                  kersize=ksize,
			                  inp_chan_sz=in_n_channel,
			                  blockidx=blockidx)
		out_n_channel = fnum

	# -------------------------------------------------------------------Finite POOLS
	elif layer_name_str == 'klavgpool':
		ksize = int(layer_opts['r'])
		stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		layer = KLAvgPool(spsize=ksize,stride=stride,pad=pad,blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str == 'glklavgpool':
		layer = KLAvgPoolGL(blockidx=blockidx)
		out_n_channel = in_n_channel
		out_icnum = 1
	# -------------------------------------------------------------------Finite Activations
	elif layer_name_str == 'lnorm':
		isstoch = bool(layer_opts['s']=='1') if 's' in  layer_opts.keys() else False
		regulated = bool(layer_opts['reg'] == '1') if 'reg' in layer_opts.keys() else False
		layer = LNorm(isstoch=isstoch,blockidx=blockidx,isregulated=regulated)
		out_n_channel = in_n_channel
	elif layer_name_str == 'l2norm':
		layer = L2Norm()
		out_n_channel = in_n_channel
	elif layer_name_str == 'l2lpnorm':
		# Normalizes the input(l2) and outputs the log probs
		layer = L2LogProb(blockidx=blockidx)
		out_n_channel = in_n_channel
	elif layer_name_str== 'kdrop':
		rate = float(layer_opts['p'])
		layer = KLDrop(rate,blockidx=blockidx)
		out_n_channel = in_n_channel
	# -------------------------------------------------------------------Input Transformers
	elif layer_name_str == 'inplog':
		layer = Inp2Log()
		out_n_channel = in_n_channel
	# -------------------------------------------------------------------Conv Equipment

	elif layer_name_str == 'conv':
		ksize = int(layer_opts['r'])
		fnum = int(layer_opts['f'])
		bias = bool(int(layer_opts['bias']))
		stride = int(layer_opts['stride'] if 'stride' in layer_opts.keys() else 1)
		pad = layer_opts['pad']
		pad = evalpad(pad,ksize)
		layer = nn.Conv2d(in_channels=in_n_channel,
						  out_channels=fnum,
						  kernel_size=ksize,
						  stride=stride,
						  padding=pad,
						  bias=bias)
		out_n_channel = fnum
	elif layer_name_str == 'relu':
		layer = nn.ReLU()
		out_n_channel = in_n_channel
	elif layer_name_str == 'sigmoid':
		layer = nn.Sigmoid()
		out_n_channel = in_n_channel
	elif layer_name_str == 'lsigmoid':
		layer = nn.LogSigmoid()
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
	elif layer_name_str =='dropout':
		prob = float(layer_opts['p'])
		layer = nn.Dropout(prob)
		out_n_channel = in_n_channel
	elif layer_name_str =='bn':

		layer = nn.BatchNorm2d(in_n_channel)
		out_n_channel = in_n_channel
	else:
		raise(Exception('Undefined Layer: ' + layer_name_str))
	if out_n_channel == -1:
		raise('Output Channel Num not assigned in :' + layer_name_str)

	blockidx_dict[layer_name_str] = blockidx_dict[layer_name_str]+1
	return layer, out_n_channel,out_icnum
