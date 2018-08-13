from experiment.Experiment import Experiment_
from optstructs import *
from typing import Tuple,List,Dict
from trainvalid.lr_schedulers import *
from torch.nn.modules import NLLLoss
from models.klmodels import *
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
		datasets = ['cifar10', 'cifar100']
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

class FiniteReLU(Experiment_):
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
class InitializationLogVsSphere(Experiment_):
	''' Effects of initialization in KL-convs in both ReLU Models and Sigmoidal Models

	Model Category 1: ReLU Models
	Model Category 2: Sigmoidal Models
	'''


class InitializationVsDepth(Experiment_):
	''' Check the choice of initialization is dependent on depth'''


class InitialCoefExperiment(Experiment_):
	''' Experiment with different initial coeficients'''