from experiment.Experiment import Experiment_
from optstructs import *
from typing import Tuple,List,Dict
from trainvalid.lr_schedulers import *
from torch.nn.modules import NLLLoss
from models.klmodels import *
import layers.klmodules as M
from data.datasetutils import *

class MAP(Experiment_):
	''' Experimenting stochastic gradients in all layers:

		Model category 1: KL ReLU models with stochastic LNORM
		Model category 2: KL Sigmoid models with stochastic LNORM
	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=500,
								 batchsz=128,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar10', 'cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.Hello_Map_variables]:
				for reg_coef in [1/50000]:#[x/10 for x in range(11)]:
					model_opt, optim_opt = model(dataopt, reg_coef=reg_coef,lr=0.01,init_coef=0.01)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)

		return opt_list

	def Hello_Map(self,data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{},reg:0'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:1,sampopt:2'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'map|r:3,f:64,pad:same,bias:1,stride:1,{}'.format(convparam) + d# + nl + d

		model_string += 'map|r:3,f:64,pad:same,bias:1,stride:1,{}'.format(convparam) + d #+ nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'map|r:3,f:128,pad:same,bias:1,stride:1,{}'.format(convparam) + d # + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'map|r:3,f:128,pad:same,bias:1,stride:1,{}'.format(convparam) + d #+ nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'map|r:3,f:128,pad:same,bias:1,stride:1,{}'.format(convparam) + d#   + nl + d
		model_string += 'klavgpool|r:3,f:64,pad:same,stride:2,bias:1' + d

		model_string += 'map|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,stride:1,islast:1,{}'.format(
			convparam)  + d# + nl + d
		model_string += 'klavgpool|r:2,f:32,pad:valid,stride:1,bias:1' + d# + nl + d

		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[ BintoLogFSD]
		lr_sched = constant_lr(init_lr=1,step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=8,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.0,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss= NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def Hello_Map_variables(self,data_opts: DataOpts,
							reg_coef=0,
							init_coef=0.0001,
							lr=1
							) -> Tuple[NetOpts, OptimOpts]:


		model_string = ''
		d = '->'
		finish = 'fin'
		samp = 'sample' + d
		convparam = 'param:log,coef:{}'.format(str(init_coef))
		convparamend = 'param:log,coef:{}'.format(str(init_coef))



		# model_string += samp
		model_string += 'map|r:3,f:2,icnum:32,pad:same,bias:0,stride:1,{}'.format(convparam) + d
		model_string += 'klavgpool|r:3,pad:same,stride:2' + d

		model_string += samp
		model_string += 'map|r:3,f:2,icnum:64,pad:same,bias:0,stride:1,{}'.format(convparam) + d
		model_string += 'klavgpool|r:3,pad:same,stride:2' + d

		model_string += samp
		model_string += 'map|r:3,f:2,icnum:64,pad:same,bias:0,stride:1,{}'.format(convparam) + d
		model_string += 'klavgpool|r:3,pad:same,stride:2' + d

		model_string += samp
		model_string += 'map|r:3,f:2,icnum:64,pad:same,bias:0,stride:1,{}'.format(convparam) + d
		model_string += 'klavgpool|r:3,pad:same,stride:2' + d

		model_string += samp
		model_string += 'map|r:3,f:2,icnum:64,pad:same,bias:0,stride:1,{}'.format(convparamend) + d
		model_string += 'klavgpool|r:3,pad:same,stride:2' + d

		model_string += samp
		model_string += 'map|r:1,f:10,icnum:1,pad:same,bias:0,stride:1,{}'.format(convparam) + d
		# model_string += 'glklavgpool|r:32,pad:valid,stride:1' + d



		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSDFact]
		lr_sched = discrete_exp_decay_lr(init_lr=1, step=2, exp_decay_perstep=0)

		''' Net Options'''
		divgreg= True
		if (reg_coef!=0)!=divgreg: Warning("Check regularizer options")
		netdict = dict(exact=False, divgreg=divgreg, reg_coef=reg_coef, reg_mode='mdivg')
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',2),("icnum",3)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )

		opts_optim = OptimOpts(lr=lr,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )

		'''Optimizer Options'''

		return opts_net, opts_optim
	def Hello_Map_onelayerv3(self,data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		''' 20 percent acc'''
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{},reg:0'.format(isnormstoch)
		convparam = 'param:log,stoch:0,isrelu:{},coef:1,sampopt:2'.format(isrelu)
		convparamsigm = 'param:logdirichunif,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'map|r:5,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d #  + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d
		model_string += 'map|r:5,f:64,pad:same,bias:1,stride:1,{}'.format(convparam) + d  # + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d
		model_string += 'map|r:5,f:64,pad:same,bias:1,stride:1,{}'.format(convparam) + d  # + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d
		model_string += 'map|r:3,f:64,pad:same,bias:1,stride:1,{}'.format(convparam) + d  # + nl + d
		model_string += 'glklavgpool|r:32,f:128,pad:valid,stride:1,bias:1' + d
		model_string += 'map|r:1,f:10,pad:same,bias:1,stride:1,{}'.format(convparam) + d  # + nl + d

		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSD]
		lr_sched = constant_lr(init_lr=1, step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=8,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def imgenerator(self,data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		isrelu=True
		isnormstoch=False
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{},reg:0'.format(isnormstoch)
		nlreg = 'lnorm|s:{},reg:1'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:2'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:1,f:3,pad:same,bias:1,stride:1,{}'.format(convparamsigm) + d + 'sigmoid'+ d
		model_string += 'klconvb|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparamsigm) + d + nl + d
		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d   + nl + d
		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d

		#model_string += 'klavgpool|r:3,f:64,pad:valid,stride:2,bias:1' + d
		model_string += 'klconv|r:2,f:' + str(data_opts.classnum) + ',pad:valid,bias:0,stride:1,{}'.format(
			convparam)  + d + nl + d

		#model_string += 'klavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d# + nl + d

		model_string += finish

		'''Data OPTs'''
		data_transforms =[]# BintoLogFSD]
		'''LR SCHED'''
		lr_sched = constant_lr(init_lr=1,step=30, exp_decay_perstep=1)
		''' Net Options'''
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def markov(self,data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		isrelu=True
		isnormstoch=False
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{},reg:0'.format(isnormstoch)
		nlreg = 'lnorm|s:{},reg:1'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:2'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'nfmarkov|r:3,f:64 ,pad:same,bias:0,stride:1'+d
		model_string += 'klavgpool|r:4,f:32,pad:same,stride:2,bias:1' + d# + nl + d
		model_string += 'nfmarkov|r:3,f:64 ,pad:same,bias:0,stride:1' + d
		model_string += 'klavgpool|r:4,f:32,pad:same,stride:2,bias:1' + d  # + nl + d
		model_string += 'nfmarkov|r:3,f:64 ,pad:same,bias:0,stride:1' + d
		model_string += 'klavgpool|r:4,f:32,pad:same,stride:2,bias:1' + d  # + nl + d
		model_string += 'nfmarkov|r:4,f:' + str(data_opts.classnum) + ',pad:valid,bias:0,stride:1,{}'.format(
			convparam)  + d

		#model_string += 'klavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d# + nl + d

		model_string += finish

		'''Data OPTs'''
		data_transforms =[]# BintoLogFSD]
		'''LR SCHED'''
		lr_sched = constant_lr(init_lr=1,step=30, exp_decay_perstep=1)
		''' Net Options'''
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.5,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim


class Synthetic_PMaps(Experiment_):
	''' Experimenting :
	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=100,
								 batchsz=100,
		                         batchsz_val=128,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['synthetic']
		for dataset in datasets:
			inputsz =20
			outputsz =5# inputsz
			numsample = [1000,1000]
			dataopt = DataOpts(dataset,inputsz,outputsz,numsample)
			for model in [self.Map]:
				for isrelu in [True]:
					model_opt, optim_opt = model(dataopt,inputsz,outputsz,isrelu=isrelu)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)

		return opt_list

	def Map(self,data_opts: DataOpts,inputsz,outputsz, isrelu=True, isnormstoch=False,reg_coef=1/20) -> Tuple[NetOpts, OptimOpts]:
		exact = False
		layerstr = 'map'
		d = '->'
		finish = 'fin'
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{},reg:0'.format(isnormstoch)
		samp = 'sample' + d
		convparam = 'param:log,stoch:0,isrelu:{},coef:0,sampopt:2,exact:0'.format(isrelu)
		convparamend = 'param:log,stoch:0,isrelu:{},coef:0.0001,sampopt:2,exact:0'.format(isrelu)
		#model_string += '{}|r:1,f:1000,pad:same,bias:1,stride:1,{}'.format(layerstr,convparam) + d  # + nl + d
		#model_string += samp
		model_string += '{}|r:1,f:10,icnum:1,pad:same,bias:0,stride:1,{}'.format(layerstr,convparam) + d  # + nl + d
		model_string += samp
		model_string += '{}|r:1,f:10,icnum:1,pad:same,bias:0,stride:1,{}'.format(layerstr,convparam) + d  # + nl + d
		model_string += samp
		model_string += ('{}|r:1,f:' + str(data_opts.classnum) + ',icnum:1,pad:valid,bias:0,stride:1,islast:1,{}').format(
			layerstr,convparam)  + d# + nl + d
		# model_string += samp
		#model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d# + nl + d

		model_string += finish

		netdict = dict(exact=exact, divgreg=reg_coef !=0, reg_coef=reg_coef, reg_mode='mdivg')
		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms =[]
		lr_sched = constant_lr(init_lr=1,step=30, exp_decay_perstep=0)
		''' Net Options'''
		opts_net = NetOpts(model_string,
						   input_channelsize=dict([('chansz',inputsz),("icnum",1)]),
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=3,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.0,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss= NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def imgenerator(self,data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		isrelu=True
		isnormstoch=False
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{},reg:0'.format(isnormstoch)
		nlreg = 'lnorm|s:{},reg:1'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:2'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:1,f:3,pad:same,bias:1,stride:1,{}'.format(convparamsigm) + d + 'sigmoid'+ d
		model_string += 'klconvb|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparamsigm) + d + nl + d
		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d   + nl + d
		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:128,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:32,pad:same,bias:1,stride:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:32,pad:same,stride:2,bias:1' + d

		#model_string += 'klavgpool|r:3,f:64,pad:valid,stride:2,bias:1' + d
		model_string += 'klconv|r:2,f:' + str(data_opts.classnum) + ',pad:valid,bias:0,stride:1,{}'.format(
			convparam)  + d + nl + d

		#model_string += 'klavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d# + nl + d

		model_string += finish

		'''Data OPTs'''
		data_transforms =[]# BintoLogFSD]
		'''LR SCHED'''
		lr_sched = constant_lr(init_lr=1,step=30, exp_decay_perstep=1)
		''' Net Options'''
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def markov(self,data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		isrelu=True
		isnormstoch=False
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{},reg:0'.format(isnormstoch)
		nlreg = 'lnorm|s:{},reg:1'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:2'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'nfmarkov|r:3,f:64 ,pad:valid,bias:0,stride:1'+d
		model_string += 'nfmarkov|r:3,f:128 ,pad:valid,bias:0,stride:1'+d
		model_string += 'nfmarkov|r:3,f:128 ,pad:valid,bias:0,stride:1'+d
		model_string += 'nfmarkov|r:3,f:64 ,pad:valid,bias:0,stride:1'+d
		model_string += 'nfmarkov|r:3,f:64 ,pad:valid,bias:0,stride:1' + d
		model_string += 'nfmarkov|r:3,f:128 ,pad:valid,bias:0,stride:1' + d
		model_string += 'nfmarkov|r:3,f:64 ,pad:valid,bias:0,stride:1' + d
		model_string += 'nfmarkov|r:5,f:' + str(data_opts.classnum) + ',pad:valid,bias:0,stride:1,{}'.format(
			convparam)  + d

		#model_string += 'klavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d# + nl + d

		model_string += finish

		'''Data OPTs'''
		data_transforms =[]# BintoLogFSD]
		'''LR SCHED'''
		lr_sched = constant_lr(init_lr=1,step=30, exp_decay_perstep=1)
		''' Net Options'''
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.5,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim


class BaseLines(Experiment_):
	def collect_opts(self):
		opt_list=[]
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=100,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar10','cifar100']
		models = ['quick_cifar','nin_caffe','vgg']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in models:
				model_opt,optim_opt = self.get_model_opts(model,dataopt)
				opt = allOpts(model,netopts=model_opt,optimizeropts=optim_opt,epocheropts=epocheropt,dataopts=dataopt)
				opt_list.append(opt)

		return opt_list


class Stochastic(Experiment_):
	''' Experimenting stochastic gradients in all layers:

		Model category 1: KL ReLU models with stochastic LNORM
		Model category 2: KL Sigmoid models with stochastic LNORM
	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=100,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [finite_nin_caffe,finite_vgg,finite_quick_cifar]:
				for isrelu in [False,False]:
					model_opt, optim_opt = model(dataopt,isrelu=isrelu)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)
		return opt_list


class Sigmoid(Experiment_):
	'''Experimenting Sigmoid vs ReLU in CNNs and FCNNs
	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=100,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar10', 'cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [finite_vgg,finite_quick_cifar,finite_nin_caffe]:
				for isrelu in [False]:
					model_opt, optim_opt = model(dataopt,isrelu=isrelu)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)
		return opt_list




class FiniteReLUStochGrad(Experiment_):
	'''Experimenting Sigmoid vs ReLU in CNNs and FCNNs
	'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=100,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar10', 'cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [finite_nin_caffe,finite_quick_cifar,finite_vgg]:
				for isrelu in [True]:
					model_opt, optim_opt = model(dataopt,isrelu=isrelu,isnormstoch=True)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)
		return opt_list


class FiniteDropout(Experiment_):
	'''The dropouts are replaced with a sampler layer and a consequent I-KLD'''
	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=100,
								 shuffledata=True,
								 numworkers=1,
								 gpu=True)
		datasets = ['cifar10', 'cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [finite_nin_caffe,finite_quick_cifar,finite_vgg]:
				for isrelu in [True]:
					model_opt, optim_opt = model(dataopt,isrelu=isrelu,isnormstoch=True)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)
		return opt_list
	def finite_nin_dropout(self,data_opts):
		model_string = ''
		isrelu = '1'
		isnormstoch = '0'
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:4'.format(isrelu)
		convparamsigmoid = 'param:logdirich,stoch:1,isrelu:{},coef:4'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'klconvb|r:5,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:1,f:160,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		# model_string += 'dropout|p:0.5' + d
		model_string += 'klconv|r:5,f:192,pad:same,bias:1,{}'.format(convparamsigmoid) + d + nl + d
		model_string += 'klconv|r:1,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:1,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		# model_string += 'dropout|p:0.5' + d
		model_string += 'klconv|r:3,f:192,pad:same,bias:1,{}'.format(convparamsigmoid) + d + nl + d
		model_string += 'klconv|r:1,f:192,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:0,{}'.format(
			convparam) + d + nl + d
		model_string += 'klavgpool|r:7,f:32,pad:valid,stride:2,bias:1' + d + nl + d

		model_string += finish

		'''Data OPTs'''
		data_transforms = []
		'''LR SCHED'''
		lr_sched = discrete_exp_decay_lr(init_lr=1, step=30, exp_decay_perstep=1)
		''' Net Options'''
		opts_net = NetOpts(model_string,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim


class ICML19(Experiment_):
	'''Experimenting Sigmoid vs ReLU in CNNs and FCNNs
	'''

	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=128,
								 shuffledata=True,
								 numworkers=2,
								 gpu=True)
		datasets = [ 'cifar100','cifar10']
		for model in [self.finite_vgg_double,self.vgg_real_noBN_noDO]:
			for dataset in datasets:
				dataopt = DataOpts(dataset)

				comps = 0
				model_opt, optim_opt = model(dataopt, comps + 1)
				opt = allOpts(model.__name__ + 'indpt_compt: ' + str(comps), netopts=model_opt,
							  optimizeropts=optim_opt, epocheropts=epocheropt,
							  dataopts=dataopt)
				opt_list.append(opt)
		return opt_list

	def finite_vgg(self, data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:4'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:0,coef:4'
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:3,f:32,icnum:2,pad:same,bias:1,{}'.format(convparamsigm) + d + nl + d #64
		model_string += 'klconv|r:3,f:32,icnum:2,pad:same,bias:1,{}'.format(convparam) + d + nl + d#64
		model_string += 'klavgpool|r:2,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:128,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d #128
		model_string += 'klconv|r:3,f:128,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d #128
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:256,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d #256
		model_string += 'klconv|r:3,f:256,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#256
		model_string += 'klconv|r:3,f:256,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#256
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # 512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:1,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'dropout|p:0.5' + d
		model_string += 'klconv|r:1,f:' + str(data_opts.classnum) + ',icnum:1,pad:valid,bias:0,{}'.format(convparam) + d
		model_string += nl + d
		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d #  + nl + d
		model_string += finish
		'''DATA OPTS'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSDFact]
		lr_sched = constant_lr(init_lr=1, step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=2,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def finite_vgg_double(self, data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:5'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:1,coef:5'
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:3,f:64,icnum:1,pad:same,bias:1,{}'.format(convparamsigm) + d + nl + d #64
		model_string += 'klconv|r:3,f:64,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#64
		model_string += 'klavgpool|r:2,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:256,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d #128
		model_string += 'klconv|r:3,f:256,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d #128
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d #256
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#256
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#256
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # 512
		model_string += 'klconv|r:3,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d#512
		model_string += 'klavgpool|r:2,pad:valid,stride:2,bias:1' + d

		model_string += 'klconv|r:1,f:512,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d  # + 'dropout|p:0.5' + d
		model_string += 'klconv|r:1,f:' + str(data_opts.classnum) + ',icnum:1,pad:valid,bias:0,{}'.format(convparam) + d
		model_string += nl + d
		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d #  + nl + d
		model_string += finish
		'''DATA OPTS'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSDFact]
		lr_sched = constant_lr(init_lr=1, step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=2,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def real_quick_cifar(self, data_opts: DataOpts, indptcomps, isrelu=False, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'relu'
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:5,f:32,pad:same,bias:1' + d
		model_string += 'maxpool|r:3,f:32,pad:same,stride:2,bias:1' + d
		model_string += 'conv|r:5,f:64,pad:same,bias:1' + d + nl + d
		model_string += 'avgpool|r:3,f:32,pad:same,stride:2,bias:1' + d
		model_string += 'conv|r:4,f:64,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:7,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d + 'lnorm|s:0' + d
		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])]
		lr_sched = exp_decay_lr(init_lr=0.01, step=20, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=True,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=2e-5,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def finite_quick_cifar(self, data_opts: DataOpts, indptcomps, isrelu=False, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:log,stoch:0,isrelu:{},coef:1'.format(isrelu)
		convparam = 'param:log,stoch:0,isrelu:{},coef:1'.format(False)
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:5,f:32,icnum:{},pad:same,bias:1,{}'.format(str(indptcomps), convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:5,f:32,icnum:{},pad:same,bias:1,{}'.format(str(indptcomps), convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:4,f:64,icnum:{},pad:same,bias:1,{}'.format(str(indptcomps), convparam) + d + nl + d

		model_string += 'klconv|r:7,icnum:{},f:'.format(str(indptcomps)) + str(
			data_opts.classnum) + ',pad:valid,bias:1,{}'.format(
			convparam) + d + nl + d
		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d  # + nl + d
		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSD]
		lr_sched = constant_lr(init_lr=.001, step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True)
		opts_net = NetOpts(model_string,
						   input_channelsize=8,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def finite_quick_cifar_single_72p(self, data_opts: DataOpts, idcomp, isrelu=True, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:5,f:16,icnum:2,pad:same,bias:1,{}'.format(convparamsigm) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:7,f:64,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:4,f:64,icnum:1,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:3,icnum:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,{}'.format(
			convparam) + d + nl + d

		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d  # + nl + d
		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSDFact]
		lr_sched = exp_decay_lr(init_lr=1, step=10, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=2,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def finite_quick_cifar_single_cmp(self, data_opts: DataOpts, idcomp, isrelu=True, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:log,stoch:0,isrelu:{},coef:1'.format(isrelu)
		convparamsigm = 'param:log,stoch:0,isrelu:0,coef:1'.format(isrelu)


		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:5,f:2,icnum:16,pad:same,bias:1,{}'.format(convparamsigm) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:5,f:4,icnum:16,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' +d

		model_string += 'klconv|r:4,f:16,icnum:4,pad:same,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'klconv|r:3,icnum:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,{}'.format(
			convparam) + d + nl + d

		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d  # + nl + d
		model_string += finish

		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSDFact]
		lr_sched = discrete_exp_decay_lr(init_lr=1, step=10, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=2,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def nin_finite_working(self, data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''
#HAS KONV BIAS
		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:logdirich,stoch:0,isrelu:{},coef:4'.format(isrelu)
		convparamsigm = 'param:logdirich,stoch:0,isrelu:{},coef:4'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:5,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:80,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:5,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,icnum:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d  # + nl + d
		model_string += finish


		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSDFact]
		lr_sched = constant_lr(init_lr=1, step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=2,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim
	def nin_real(self, data_opts: DataOpts, isrelu=False, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'relu'
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:160,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:96,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		model_string += 'dropout|p:0.5' + d
		model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'avgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		model_string += 'dropout|p:0.5' + d
		model_string += 'conv|r:3,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d + nl + d
		model_string += 'avgpool|r:7,f:32,pad:valid,stride:2,bias:1' + d + 'lnorm|s:0' + d
		model_string += finish
		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = nin_caffe_lr
		lr_sched = constant_lr(init_lr=0.1)

		''' Net Options'''
		netdict = dict(exact=True, divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=True,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=.1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=1e-4,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim


	def nin_real_noDO(self, data_opts: DataOpts, isrelu=False, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'relu'
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:160,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:96,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		#model_string += 'dropout|p:0.5' + d
		model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'avgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		#model_string += 'dropout|p:0.5' + d
		model_string += 'conv|r:3,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d + nl + d
		model_string += 'avgpool|r:7,f:32,pad:valid,stride:2,bias:1' + d + 'lnorm|s:0' + d
		model_string += finish
		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = nin_caffe_lr
		lr_sched = constant_lr(init_lr=0.1)

		''' Net Options'''
		netdict = dict(exact=True, divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=True,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=.1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=1e-4,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim
	def nin_finite_do_sphere(self, data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:sphere,stoch:0,isrelu:{},coef:1'.format(isrelu)
		convparamsigm = 'param:sphere,stoch:0,isrelu:{},coef:1'.format('0')
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:5,f:192,icnum:1,pad:same,bias:1,{}'.format( convparamsigm) + d + nl + d
		model_string += 'klconv|r:1,f:160,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d
		model_string += 'dropout|p:0.5' + d
		model_string += 'klconv|r:5,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d
		model_string += 'dropout|p:0.5' + d
		model_string += 'klconv|r:3,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,icnum:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d  # + nl + d
		model_string += finish


		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSDFact]
		lr_sched = discrete_exp_decay_lr(init_lr=1, step=50, exp_decay_perstep=math.log(10))

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=2,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.0,
							   weight_decay=1e-4,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def vgg_real(self, data_opts: DataOpts, isrelu=False, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'relu'
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:3,f:64,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.3' + d
		model_string += 'conv|r:3,f:64,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:128,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:128,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d + 'bn' + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:1,f:512,pad:same,bias:1' + d + nl + d + 'dropout|p:0.5' + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d
		model_string += 'lnorm|s:0' + d
		model_string += finish
		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = vgg_lr
		#lr_sched = constant_lr(init_lr=0.1)

		''' Net Options'''
		netdict = dict(exact=True, divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=True,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=.1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=1e-4,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim
	def vgg_real_noBN_noDO(self, data_opts: DataOpts, isrelu=False, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'relu'
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:3,f:64,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.3' + d
		model_string += 'conv|r:3,f:64,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:128,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:128,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:256,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d #+ 'bn' + d + 'dropout|p:0.4' + d
		model_string += 'conv|r:3,f:512,pad:same,bias:1' + d + nl + d #+ 'bn' + d
		model_string += 'maxpool|r:2,pad:valid,stride:2,bias:1' + d
		model_string += 'conv|r:1,f:512,pad:same,bias:1' + d + nl + d #+ 'dropout|p:0.5' + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d
		model_string += 'lnorm|s:0' + d
		model_string += finish
		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		#lr_sched = vgg_lr
		lr_sched = constant_lr(init_lr=1)

		''' Net Options'''
		netdict = dict(exact=True, divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=True,
						   weightinit=lambda x: x.normal_(0, 0.03),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=.001,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=1e-4,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim


class NIN(Experiment_):
	'''Experimenting Sigmoid vs ReLU in CNNs and FCNNs
	'''

	def collect_opts(self):
		opt_list = []
		epocheropt = EpocherOpts(self.save_results,
								 epochnum=150,
								 batchsz=128,
								 shuffledata=True,
								 numworkers=2,
								 gpu=True)
		datasets = ['cifar100']
		for dataset in datasets:
			dataopt = DataOpts(dataset)
			for model in [self.nin_finite]:
				for isrelu in [True]:
					model_opt, optim_opt = model(dataopt)
					opt = allOpts(model.__name__, netopts=model_opt, optimizeropts=optim_opt, epocheropts=epocheropt,
								  dataopts=dataopt)
					opt_list.append(opt)
		return opt_list

	def nin_real(self, data_opts: DataOpts, isrelu=False, isnormstoch=False) -> Tuple[NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'relu'
		d = '->'
		finish = 'fin'
		model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:160,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:96,pad:same,bias:1' + d + nl + d
		model_string += 'maxpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		model_string += 'dropout|p:0.5' + d
		model_string += 'conv|r:5,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'avgpool|r:3,f:32,pad:valid,stride:2,bias:1' + d
		model_string += 'dropout|p:0.5' + d
		model_string += 'conv|r:3,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:192,pad:same,bias:1' + d + nl + d
		model_string += 'conv|r:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1' + d + nl + d
		model_string += 'avgpool|r:7,f:32,pad:valid,stride:2,bias:1' + d + 'lnorm|s:0' + d
		model_string += finish
		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
		lr_sched = nin_caffe_lr

		''' Net Options'''
		netdict = dict(exact=True, divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=3,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=True,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=2e-3,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=1e-4,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim

	def nin_finite(self, data_opts: DataOpts, isrelu=True, isnormstoch=False) -> Tuple[
		NetOpts, OptimOpts]:
		model_string = ''

		isrelu = booltostr(isrelu)
		isnormstoch = booltostr(isnormstoch)
		nl = 'lnorm|s:{}'.format(isnormstoch)
		convparam = 'param:log,stoch:0,isrelu:{},coef:15'.format(isrelu)
		d = '->'
		finish = 'fin'
		model_string += 'klconv|r:5,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:80,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:5,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:192,icnum:1,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klavgpool|r:3,pad:same,stride:2,bias:1' + d

		model_string += 'klconv|r:3,f:48,icnum:4,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,f:96,icnum:2,pad:same,bias:1,{}'.format( convparam) + d + nl + d
		model_string += 'klconv|r:1,icnum:1,f:' + str(data_opts.classnum) + ',pad:valid,bias:1,{}'.format(convparam) + d + nl + d
		model_string += 'glklavgpool|r:4,f:32,pad:valid,stride:2,bias:1' + d  # + nl + d
		model_string += finish


		'''Data OPTs'''
		'''LR SCHED'''
		data_transforms = [BintoLogFSD]
		lr_sched = constant_lr(init_lr=1, step=30, exp_decay_perstep=1)

		''' Net Options'''
		netdict = dict(exact=True,divgreg=False)
		opts_net = NetOpts(model_string,
						   input_channelsize=8,
						   inputspatszvalidator=lambda x: x == 32,
						   data_transforms=data_transforms,
						   classicNet=False,
						   weightinit=lambda x: x.normal_(0, 0.05),
						   biasinit=lambda x: x.zero_(),
						   customdict=netdict
						   )
		opts_optim = OptimOpts(lr=1,
							   lr_sched_lambda=lr_sched,
							   type='SGD',
							   momentum=0.9,
							   weight_decay=0,
							   dampening=0,
							   nestrov=False,
							   loss=NLLLoss(reduce=False)
							   )
		'''Optimizer Options'''

		return opts_net, opts_optim
