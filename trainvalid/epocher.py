
import torch.nn as nn
from  netparsers.staticnets import StaticNet
from optstructs import allOpts
from layers.klmodules import MyModule
from torch.optim import Optimizer
from resultutils.resultstructs import *
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import *
from torch.nn.modules import Module
class Epocher(object):
	def __init__(self,opts:allOpts):

		super(Epocher, self).__init__()
		self.opts = opts
		self.model = self.create_model_module(opts) #type:MyModule
		self.trainloader , self.testloader = self.create_data_set(opts)
		self.optimizer = self.create_optimizer(opts,self.model)
		self.results=None
	def create_data_set(self, opts):
		if opts.dataopts.datasetname is 'cifar10':
			train_loader, test_loader = self.opts.dataopts.get_cifar10(self.opts)
		elif opts.dataopts.datasetname is 'cifar100':
			train_loader, test_loader = self.opts.dataopts.get_cifar100(self.opts)
		else:
			raise ('Dataset Loader not found: ' + self.opts.dataopts.datasetname)
		return train_loader, test_loader
	def reinstantiate_model(self):
		self.model = self.create_model_module(self.opts)  # type:MyModule
		self.optimizer = self.create_optimizer(self.opts, self.model)
		self.results = None
	def create_model_module(self,opts) -> StaticNet:
		module = StaticNet(opts.netopts.modelstring,
		                   opts.dataopts.channelsize,
		                   weightinit=opts.netopts.weightinit,
		                   biasinit=opts.netopts.biasinit)
		return module

	def create_optimizer(self,opts, model: Module):
		optimopts = opts.optimizeropts
		if opts.epocheropts.gpu:
			device = torch.device("cpu")
			model = model.to(device=device)

		optim = globals()[optimopts.type](model.parameters(),
		                                  lr=1,
		                                  momentum=optimopts.momentum,
		                                  weight_decay=optimopts.weight_decay,
		                                  dampening=optimopts.dampening,
		                                  nesterov=optimopts.nestrov)
		opts.optimizeropts.lr_sched = LambdaLR(optim, opts.optimizeropts.lr_sched_lambda, last_epoch=-1)
		return optim

	def run_epoch(self,prefixprint:str,epoch)->dict:
		run_loss = 0.0
		corrects = 0
		totalsamples = 0
		self.model.train()
		# TODO: Train on batches
		for batch_n,data in enumerate(self.trainloader):
			inputs, labels = data
			inputs, labels = inputs.to(self.opts.epocheropts.device),labels.to(self.opts.epocheropts.device)
			self.optimizer.zero_grad()
			output = self.model(inputs)
			output = output.view(-1,self.opts.dataopts.classnum)
			# Loss Components
			log_prob = self.model.get_log_prob()
			regularizer = self.model.get_reg_vals()
			loss = self.opts.optimizeropts.loss(output,labels)
			# End Loss Components
			raw_loss = loss.mean()
			loss = loss+ (1-(-loss).exp()).detach()*log_prob + regularizer
			loss = loss.mean()
			loss.backward()

			self.optimizer.step()
			loss = raw_loss
			#expected_logprob = log_prob.mean()
			#expected_regularizer = regularizer.mean()
			#TODO: Print Batch Statistics
			predlab = torch.argmax(output, 1, keepdim=False)
			run_loss += loss.item()
			accthis = (predlab == labels).sum().item()
			corrects += accthis
			totalsamples += labels.size(0)
			train_acc = corrects/totalsamples
			train_loss = (run_loss/(batch_n+1))
			print(' '+ prefixprint +':' 
				  'batch: %d '%(batch_n) +
				  ' train_loss: %.4f'% train_loss +
			      ' train accuracy: %.2f'% (train_acc*100),end="\r"
			      )
		scalar_dict = self.model.get_scalar_dict()
		print(' '+ prefixprint + ':'
		                    'batch: %d ' % (batch_n) +
		      ' train_loss: %.4f' % train_loss +
		      ' train accuracy: %.2f' % (train_acc * 100), end=" "
		      )

		# TODO: Evaluate Test
		val_run_loss = 0
		totalsamples = 0
		val_corrects = 0
		self.model.eval()
		for batch_n,data in enumerate(self.testloader):
			with torch.set_grad_enabled(False):
				inputs, labels = data
				inputs, labels = inputs.to(self.opts.epocheropts.device),labels.to(self.opts.epocheropts.device)
				self.optimizer.zero_grad()
				output = self.model(inputs)
				output = output.view(-1, self.opts.dataopts.classnum)
				#TODO: Print Batch Statistics
				predlab = torch.argmax(output, 1, keepdim=False)
				accthis = (predlab == labels).sum().item()
				val_corrects += accthis
				totalsamples += labels.size(0)
				val_acc = val_corrects/totalsamples
				loss = self.opts.optimizeropts.loss(output,labels).mean()
				val_run_loss += loss.item()
				val_avg_loss = val_run_loss/(batch_n+1)

		print(' '+ prefixprint + ':' +
		      ' val_loss: %.4f' % val_avg_loss +
		      ' val_acc: %.2f' % (val_acc * 100))
		resdict = dict(test_acc=val_acc*100, train_acc=train_acc*100, test_loss=val_avg_loss, train_loss=train_loss)
		resdict.update(scalar_dict)
		return resdict

	def run_many_epochs(self,path:str,save_result):
		self.model.to(self.opts.epocheropts.device)
		self.results = ResultStruct(path)
		for epoch in range(self.opts.epocheropts.epochnum):
			prefixtext = 'Epoch %d' % epoch
			epochres = self.run_epoch(prefixtext,epoch)

			self.results.add_epoch_res_dict(epochres,epoch,save_result)
			self.opts.optimizeropts.lr_sched.step()
		self.model.to(torch.device('cpu'))
		return self.results


