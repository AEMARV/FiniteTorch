
import torch.nn as nn
from  netparsers.staticnets import StaticNet,DenseNet,CompositeNet
from optstructs import allOpts
from layers.klmodules import MyModule
from torch.optim import Optimizer
from resultutils.resultstructs import *
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import *
from torch.nn.modules import Module
from torchvision.utils import *
from layers.pmaputils import *
from torch.utils.data.dataloader import DataLoader
from layers.klmodules import *
import timeit
import random
import tkinter as tk
class Epocher(object):
	def __init__(self,opts:allOpts):

		super(Epocher, self).__init__()
		self.opts = opts
		self.trainloader, self.testloader = self.create_data_set(opts) # type:DataLoader
		self.model = self.create_model_module(opts,sample_data=self.trainloader.__iter__().next()[0]) #type:MyModule
		self.optimizer = self.create_optimizer(opts,self.model)
		self.results=None
		self.path=None

	def create_data_set(self, opts):

		train_loader, test_loader = self.opts.dataopts.get_loaders(self.opts)

		return train_loader, test_loader

	def reinstantiate_model(self):
		self.model = self.create_model_module(self.opts, sample_data=None)  # type:MyModule
		self.optimizer = self.create_optimizer(self.opts, self.model)
		self.results = None

	def create_model_module(self,opts,sample_data=None) -> StaticNet:
		#  TODO: Static Net is replaced by Composite Net
		module = StaticNet(opts.netopts.modelstring,
		                   opts.netopts.input_channelsize,
		                   weightinit=opts.netopts.weightinit,
		                   biasinit=opts.netopts.biasinit,
		                   sample_data=sample_data)
		return module

	def create_optimizer(self,opts, model: Module):
		optimopts = opts.optimizeropts
		if opts.epocheropts.gpu:
			device = torch.device("cpu")
			model = model.to(device=device)

		optim = globals()[optimopts.type](model.parameters(),
		                                  lr=optimopts.lr,
		                                  momentum=optimopts.momentum,
		                                  weight_decay=optimopts.weight_decay,
		                                  dampening=optimopts.dampening,
		                                  nesterov=optimopts.nestrov)
		opts.optimizeropts.lr_sched = LambdaLR(optim, opts.optimizeropts.lr_sched_lambda, last_epoch=-1)
		return optim

	def order_batch_by_label(self,batch,labels):
		_,indices = labels.sort(dim=0)
		batch = batch[indices,:,:,:]
		return batch

	def logical_index(self,batch:Tensor,booleanind):
		if booleanind.all() and booleanind.numel()==1:
			return batch
		int_index= booleanind.nonzero().squeeze()
		return batch.index_select(dim=0,index=int_index)

	def label_to_onehot(self,output:Tensor, label):
		onehot = output.new_zeros(output.size())
		label= label.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
		label = label.transpose(0,1)
		onehot.scatter_(1,label,1)
		return onehot

	def block_grad(self,paramlist:List,ind=None):
		length = paramlist.__len__()
		totalnorm = 0
		if ind is None:
			ind = random.randint(0, length-1)
		maxnorm= -1
		maxind = -1
		for i in range(length):
			if paramlist[i].grad is None: continue
			thisnorm = (paramlist[i].grad**2).sum()
			totalnorm +=thisnorm
			if thisnorm> maxnorm:
				maxnorm = thisnorm
				# if maxind!=-1 :paramlist[maxind].grad= None
				maxind = i
		return totalnorm.sqrt()

	def rnd_block_grad(self,paramlist:List,ind=None):
		length = paramlist.__len__()
		totalnorm = 0
		if ind is None:
			ind = random.randint(0, length-1)
		for i in range(length):
			if paramlist[i].grad is None:
				ind= ind-1
				continue
			totalnorm = totalnorm + paramlist[i].grad.sum()
			if i!= ind :paramlist[i].grad= None
		return

	def normalize_grad(self,paramlist:List):
		length = paramlist.__len__()

		for i in range(length):
			g = paramlist[i].grad
			paramlist[i].grad = paramlist[i].grad/ float((math.log(g.shape[1])))

	def deltac_optimization(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		usemin=False
		lprob_currect = 0
		sampler = Sampler(blockidx=-1)
		while True:
			iternum += 1
			output_temp, lp_mgh, _ = self.model(inputs,mode='delta')
			#TODO min
			output,lp_temp = sampler(output_temp,mode='delta')
			if lp_mgh is not None:
				lp_temp = self.model.accumulate_lprob(lp_temp,usemin=usemin)
				lp_mgh = self.model.accumulate_lprob_pair(lp_temp,lp_mgh,usemin=usemin)
			else:
				lp_mgh= lp_temp
			outputlabelsample = output.max(dim=1, keepdim=True)[1]

			# calc accept/reject masks
			is_oracle = (outputlabelsample.squeeze() == labels).float()
			lp_ogh = is_oracle.log()
			lp_deltac_gh = (1-lp_ogh.exp() - lp_mgh.exp()*(1-2*lp_ogh.exp())+definition.epsilon).log()
			isdeltac = lp_deltac_gh.exp() > torch.rand_like(lp_deltac_gh)
			to_be_summed = (isdeltac).squeeze().detach()

			if iternum == 1:
				ret_ldeltac = -lp_deltac_gh.mean().detach()
				ret_output = output_temp.detach()
				ret_entropy = lp_mgh.mean().detach()

			#lprob_currect += (((-lp_mgh.squeeze() * (to_be_summed.float())) / batchsz).sum()).detach()
			lossdc = (-(lp_deltac_gh[to_be_summed].float()).sum())/batchsz #+ (lp_delta_gh[to_be_summed^1].sum())/batchsz
			loss =   lossdc.sum()
			definition.hasnan(loss)
			definition.hasinf(loss)
			if to_be_summed.float().sum() >0:# 0:
				pass
				loss.sum().backward()
				#break
			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		return ret_ldeltac, ret_output, ret_ldeltac, ret_entropy,None

	def delta_optimization(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		lprob_currect = 0
		mode='delta'
		while True:
			iternum += 1
			output, lp_mgh, _ = self.model(inputs, mode=mode)
			# TODO min
			outputlabelsample = output.max(dim=1, keepdim=True)[1]

			# calc accept/reject masks
			is_oracle = (outputlabelsample.squeeze() == labels).float()
			is_model =  (lp_mgh.exp() > torch.rand_like(lp_mgh)).float()
			lp_ogh = is_oracle.log()
			lp_delta_gh = ((lp_ogh.exp()+lp_mgh.exp()-2*lp_ogh.exp()*lp_mgh.exp())+definition.epsilon).log()
			lp_deltac_gh = (1-lp_ogh.exp() - lp_mgh.exp()*(1-2*lp_ogh.exp())+definition.epsilon).log()
			isdelta = lp_delta_gh.exp() > torch.rand_like(lp_deltac_gh)
			isdeltac = isdelta^1
			to_be_summed = (isdelta).squeeze().detach()

			if iternum == 1:
				ret_ldeltac = -lp_delta_gh.mean().detach()

			#lprob_currect += (((-lp_mgh.squeeze() * (to_be_summed.float())) / batchsz).sum()).detach()
			lossdc = ((lp_delta_gh[to_be_summed].float()).sum())/batchsz #+ (lp_delta_gh[to_be_summed^1].sum())/batchsz
			loss =   lossdc.sum()
			definition.hasnan(loss)
			definition.hasinf(loss)
			if to_be_summed.float().sum() >0:# 0:
				pass
				loss.sum().backward()
				break
			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		return ret_ldeltac, ret_ldeltac, ret_ldeltac, ret_ldeltac

	def likelihoodc_optimization(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		lprob_currect = 0

		while True:
			iternum += 1
			output, lp_hgm, _ = self.model(inputs)
			output, lprob = sample_manual(output)
			output = output.log()
			lp_hgm += lprob.squeeze()
			# lp_hgm = softmin_pair(lp_hgm,lprob.squeeze())
			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)


			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle^1).squeeze().detach()

			if iternum == 1:
				ret_likelihood = loss.mean().detach()
				ret_entropy = -lp_hgm.mean().detach()

			loss =   lp_hgm[to_be_summed].sum()

			if to_be_summed.float().sum() >0:# 0:
				loss = loss / to_be_summed.sum().float()  # batchsz
				definition.hasnan(loss)
				definition.hasinf(loss)
				pass
				loss.backward()
				break

			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		return ret_likelihood,ret_entropy, ret_entropy, ret_entropy

	def likelihood_roll_optimization(self,inputs,labels):
		roll_coef= 1
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		lprob_currect = 0
		usemin= False
		sampler = Sampler(blockidx=-1)
		while True:
			iternum += 1
			output, lp_hgm, _ = self.model(inputs)
			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)
			if iternum== 1:
				roll_size= loss.new_ones(loss.size())*roll_coef
			roll_current = torch.min((-loss).exp(), roll_size)
			roll_size = roll_size -roll_current

			if iternum == 1:
				ret_likelihood = loss.mean().detach()
				ret_entropy = -lp_hgm.mean().detach()


			if roll_current.sum() >0:
				total_corrects += (-loss).exp().sum().float()
				loss = -(self.model.accumulate_lprob_pair(lp_hgm,-loss,usemin=usemin))*(roll_current.detach())
				total_samples += float(inputs.shape[0])

				loss = loss / (batchsz*roll_coef)
				definition.hasnan(loss)
				definition.hasinf(loss)
				loss.sum(). backward()

			if roll_size.sum()==0:
				break
			inputs = self.logical_index(inputs, roll_size != 0)
			labels = self.logical_index(labels, roll_size != 0)
			roll_size = self.logical_index(roll_size, roll_size != 0)
		return ret_likelihood,ret_entropy, total_corrects/total_samples, ret_entropy

	def model_stats(self,inputs,labels):
		''' Returns Lable Likelihood, generated output'''
		with torch.no_grad():
			output, logprob, _ = self.model(inputs)
			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels).mean()
		return loss, output.detach()

	def prior_variational_optimization(self, inputs, labels, coef=1):
		usemin = False
		mode = 'variational_prior'
		sampler = Sampler(blockidx=-1)
		output, lpmodel_given_h, _ = self.model(inputs, mode=mode, usemin=usemin)
		output, lp_temp = sampler(output, mode=mode)
		if lpmodel_given_h is not None:
			lp_temp = self.model.accumulate_lprob(lp_temp, usemin=usemin)
			lpmodel_given_h = self.model.accumulate_lprob_pair(lpmodel_given_h,lp_temp,usemin=usemin)
		else:
			lpmodel_given_h = lp_temp
		coef2 = (lpmodel_given_h-1).detach()
		((-lpmodel_given_h*coef2).mean() * coef).backward()

		return

	def variational_optimization(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples = 0
		total_corrects = 0
		lprob_currect = 0
		usemin= False
		sampler = Sampler(blockidx=-1)

		while True:
			iternum += 1
			output, lp_hgm, state = self.model(inputs)
			if iternum ==1:
				onehot_labels = self.label_to_onehot(output, labels)

			output, lprob = sampler(output)
			if lp_hgm is not None:
				lprob = self.model.accumulate_lprob(lprob,usemin=usemin)
				lp_hgm = self.model.accumulate_lprob_pair(lprob,lp_hgm,usemin=usemin)
			else:
				lp_hgm = lprob
			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)
			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle).squeeze().detach()
			if iternum == 1:
				ret_likelihood = loss.mean().detach()
				ret_entropy = -lp_hgm.mean().detach()
			loss = lp_hgm
			loss = -loss[to_be_summed].sum()
			total_samples += float(inputs.shape[0])
			total_corrects += to_be_summed.sum().float()
			if to_be_summed.float().sum() >0:# 0:
				loss = loss / batchsz
				definition.hasnan(loss)
				definition.hasinf(loss)
				pass
				loss.backward()
				#break

				lp_hgm_o = self.model.p_invert(state, onehot_labels)
				coef_hgmo = (lp_hgm - lp_hgm_o - 1).detach()
				loss = lp_hgm_o * coef_hgmo
				loss = -loss[to_be_summed].sum()/batchsz
				loss.backward()
			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
			onehot_labels = self.logical_index(onehot_labels, to_be_summed ^ 1)
		return ret_likelihood,ret_entropy, total_corrects/total_samples, ret_entropy

	def likelihood_optimization(self,inputs,labels,usemin=False,concentration=1.0):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		#TODO
		sampler = Sampler(blockidx=-1)
		while True:

			iternum += 1
			output_model, lp_hgm, stats = self.model(inputs,usemin=usemin,concentration=concentration)
			output, lprob = sampler(output_model,concentration=concentration)

			if lp_hgm is not None:
				lprob = self.model.accumulate_lprob(lprob,usemin=usemin)
				lp_hgm = self.model.accumulate_lprob_pair(lprob,lp_hgm,usemin=usemin)
			else:
				lp_hgm= lprob

			loss = self.opts.optimizeropts.loss(output.view(inputs.shape[0], -1), labels)
			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle).squeeze().detach()

			if iternum == 1:
				ret_output= output_model.detach()
				ret_output = ret_output-ret_output.logsumexp(dim=1,keepdim=True).detach()
				ret_likelihood = self.opts.optimizeropts.loss(ret_output.view(-1, self.opts.dataopts.classnum), labels).mean().detach()
				ret_entropy = -lp_hgm.mean().detach()
				total_samples += float(inputs.shape[0])
				total_corrects += to_be_summed.sum().float()
			if to_be_summed.float().sum() >0:
				loss = (-lp_hgm[to_be_summed]).sum()
				loss = loss / batchsz
				definition.hasnan(loss)
				definition.hasinf(loss)
				loss.backward()

			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		# print(stats['jsd'],end='')
		return ret_likelihood,ret_output, total_corrects/total_samples, ret_entropy,stats

	def likelihood_optimizationv2(self,inputs,labels,usemin=False):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		sampler = Sampler(blockidx=-1)
		while True:

			iternum += 1
			output_model, lp_hgm, stats = self.model(inputs,usemin=usemin)
			output, lprob = sampler.sample_concentrated(output_model,concentration=1)

			if lp_hgm is not None:
				lprob = self.model.accumulate_lprob(lprob,usemin=usemin)
				lp_hgm = self.model.accumulate_lprob_pair(lprob,lp_hgm,usemin=usemin)
			else:
				lp_hgm= lprob

			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)
			to_be_summed = (-loss).exp().mean() > torch.rand_like(loss)

			if iternum == 1:
				ret_output= output_model.detach()
				ret_output = ret_output-ret_output.logsumexp(dim=1,keepdim=True).detach()
				ret_likelihood = self.opts.optimizeropts.loss(ret_output.view(-1, self.opts.dataopts.classnum), labels).mean().detach()
				ret_entropy = -lp_hgm.mean().detach()
				total_samples += float(inputs.shape[0])
				total_corrects += to_be_summed.sum().float()
			if to_be_summed.float().sum() >0:
				loss = (-lp_hgm).mean()
				definition.hasnan(loss)
				definition.hasinf(loss)
				loss.backward()

			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		# print(stats['jsd'],end='')
		return ret_likelihood,ret_output, total_corrects/total_samples, ret_entropy,stats
	def intersect_optimization(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		usemin= False
		sampler = Sampler(blockidx=-1)
		while True:

			iternum += 1
			output_model, lp_hgm, stats = self.model.forward_intersect(inputs,mode='intersect')
			output, lprob = sampler(output_model,logprob_accumulate = lp_hgm, mode= 'intersect')

			lp_hgm = lprob.squeeze()

			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)
			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle).squeeze().detach()

			if iternum == 1:
				ret_output= output_model.detach()
				ret_output = ret_output-ret_output.logsumexp(dim=1,keepdim=True).detach()
				ret_likelihood = self.opts.optimizeropts.loss(ret_output.view(-1, self.opts.dataopts.classnum), labels).mean().detach()
				ret_entropy = -lp_hgm.mean().detach()
				total_samples += float(inputs.shape[0])
				total_corrects += to_be_summed.sum().float()
			if to_be_summed.float().sum() >0:
				loss = (-lp_hgm[to_be_summed]).sum()
				loss = loss / batchsz
				definition.hasnan(loss)
				definition.hasinf(loss)
				loss.backward()

			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		# print(stats['jsd'],end='')
		return ret_likelihood,ret_output, total_corrects/total_samples, ret_entropy,stats

	def EM(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		usemin= False

		iternum += 1
		output, lp_hgm, stats = self.model(inputs)
		loss = -self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)
		if lp_hgm is not None:
			lp_hgm = self.model.accumulate_lprob_pair(loss,lp_hgm,usemin=usemin)
		else:
			lp_hgm= loss

		(-lp_hgm.mean()).backward()



		return -loss.mean().detach(), output.detach(), 1, lp_hgm.mean(), stats


	def likelihood_flat_optimization(self,inputs,labels,priorcoef=1.0):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		usemin= False
		sampler = Sampler(blockidx=-1)
		print("done: ",end='',flush=True)
		while True:

			iternum += 1
			output_model, lp_hgm, stats = self.model(inputs,usemin=usemin)
			output, lprob = sampler(output_model)

			if lp_hgm is not None:
				lprob = self.model.accumulate_lprob(lprob,usemin=usemin)
				lp_hgm = self.model.accumulate_lprob_pair(lprob, lp_hgm,usemin=usemin)
			else:
				lp_hgm = lprob

			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)
			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle).squeeze().detach()

			if iternum == 1:
				ret_output= output_model.detach()
				ret_output = ret_output-ret_output.logsumexp(dim=1,keepdim=True).detach()
				ret_likelihood = self.opts.optimizeropts.loss(ret_output.view(-1, self.opts.dataopts.classnum), labels).mean().detach()
				ret_entropy = -lp_hgm.mean().detach()
				total_samples += float(inputs.shape[0])
				total_corrects += to_be_summed.sum().float()
			if to_be_summed.float().sum() >0:
				loss = (-lp_hgm[to_be_summed]).sum()
				loss = loss / batchsz
				if iternum ==1:
					loss = loss + lp_hgm.mean()*priorcoef
				definition.hasnan(loss)
				definition.hasinf(loss)
				loss.backward()

			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
			percent = 100*(1-(inputs.shape[0]/batchsz))
			print(int(percent),end='\b'*(str(int(percent)).__len__()),flush=True)

		print(int(percent), end=' ', flush=True)
		# print("JSD: ", stats['jsd'],end=' ')
		return ret_likelihood,ret_output, total_corrects/total_samples, ret_entropy,stats

	def likelihood_px_optimization(self,inputs,labels):
		batchsz = inputs.shape[0]
		iternum= 0
		total_samples= 0
		total_corrects= 0
		usemin= False
		sampler = Sampler(blockidx=-1)
		flag1= True
		while True:

			iternum += 1
			output_model, lp_hgm, stats = self.model(inputs)
			output, lprob = sampler(output_model)

			if lp_hgm is not None:
				lprob = self.model.accumulate_lprob(lprob,usemin=usemin)
				lp_hgm = self.model.accumulate_lprob_pair(lprob,lp_hgm,usemin=usemin)
			else:
				lp_hgm= lprob

			loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels)
			is_oracle = (-loss).exp() > torch.rand_like(loss)
			to_be_summed = (is_oracle).squeeze().detach()

			if iternum == 1:
				ret_output= output_model.detach()
				ret_output = ret_output-ret_output.logsumexp(dim=1,keepdim=True).detach()
				ret_likelihood = self.opts.optimizeropts.loss(ret_output.view(-1, self.opts.dataopts.classnum), labels).mean().detach()
				ret_entropy = -lp_hgm.mean().detach()
			total_samples += float(inputs.shape[0])

			if to_be_summed.float().sum() >0:
				loss = (-lp_hgm[to_be_summed].sum() + lp_hgm[to_be_summed^1].sum())/batchsz
				total_corrects += to_be_summed.sum().float()
				# loss = loss / batchsz
				definition.hasnan(loss)
				definition.hasinf(loss)
				loss.backward()
				break
			if to_be_summed.all():
				break
			inputs = self.logical_index(inputs, to_be_summed ^ 1)
			labels = self.logical_index(labels, to_be_summed ^ 1)
		# print(stats['jsd'],end='')
		return ret_likelihood,ret_output, total_corrects/total_samples, ret_entropy,stats

	def prior_optimization(self,inputs,labels,coef=1.0,mode=None,concentrate=1.0):
		if coef == 0: return
		usemin= False
		if mode is None : mode = self.opts.netopts.customdict['reg_mode']
		mode = 'likelihood'
		# mode = 'cross_entropy_unif'
		sampler = Sampler(blockidx=-1)
		output, lpmodel_given_h,_ = self.model(inputs,mode=mode,usemin=usemin,concentration=concentrate)
		output , lp_temp = sampler(output,mode=mode,concentration=concentrate)
		if lpmodel_given_h is not None:
			lp_temp = self.model.accumulate_lprob(lp_temp,usemin=usemin)
			lpmodel_given_h = self.model.accumulate_lprob_pair(lpmodel_given_h,lp_temp,usemin=usemin)
		else:
			lpmodel_given_h= lp_temp
		(lpmodel_given_h.mean()*coef).backward()


		return
	def renyi_prior(self,inputs,coef=1.0):
		sampler = Sampler(blockidx=-1)
		batchsz = inputs.shape[0]
		mode = 'likelihood'
		usemin = False
		batchsz = inputs.shape[0]
		i=0

		def sample_model(inputs):
			output, lph1, _ = self.model(inputs, usemin=False, mode=mode)
			output, lp_temp1 = sampler(output, mode=mode)
			lp_temp1 = self.model.accumulate_lprob(lp_temp1, usemin=usemin)
			lph1 = self.model.accumulate_lprob_pair(lph1, lp_temp1, usemin=usemin)
			return output,lph1

		with torch.set_grad_enabled(False):
			_, lprob_anchor = sample_model(inputs)
			for i in range(10):

						_, lprob_anchor_temp = sample_model(inputs)
						lprob_anchor = torch.max(lprob_anchor,lprob_anchor_temp)

		accept= lprob_anchor> 100.0
		while(~(accept.all())):
			_, lprob = sample_model(inputs)
			accept = (lprob_anchor + torch.rand_like(lprob).log())<= lprob
			(lprob[accept.squeeze()].sum()/batchsz).backward()
			inputs = self.logical_index(inputs,~accept)
			lprob_anchor = self.logical_index(lprob_anchor,~accept)





		def randgen(lph1):
			return torch.rand_like(lph1).log()


	def renyi_prior_MCMC(self,inputs,coef=1.0):
		sampler = Sampler(blockidx=-1)
		batchsz = inputs.shape[0]
		mode = 'likelihood'
		usemin = False
		batchsz = inputs.shape[0]
		i=0

		def sample_model(inputs):
			output, lph1, _ = self.model(inputs, usemin=False, mode=mode)
			output, lp_temp1 = sampler(output, mode=mode)
			lp_temp1 = self.model.accumulate_lprob(lp_temp1, usemin=usemin)
			lph1 = self.model.accumulate_lprob_pair(lph1, lp_temp1, usemin=usemin)
			return output,lph1
		for i in range(100):
			_, lph1 = sample_model(inputs)
			if(i==0):
				lph_anchor = lph1
				continue
			select_new = (2*(lph1-lph_anchor)) > torch.rand_like(lph1).log()
			lph_anchor[select_new] = lph1[select_new]

		lph_anchor.mean().backward()




		def randgen(lph1):
			return torch.rand_like(lph1).log()


	def run_epoch(self,prefixprint:str,epoch,path)->dict:
		run_loss = 0.0
		reg_loss = 0
		run_lprob_correct=0.0
		trytotal = 0
		corrects = 0
		totalsamples = 0
		thisNorm=0
		self.model.train()


		# TODO: Train on batches
		for batch_n,data in enumerate(self.trainloader):

			inputs, labels = data
			inputs, labels = inputs.to(self.opts.epocheropts.device),labels.to(self.opts.epocheropts.device)
			if batch_n ==0:
				fix_batch= inputs[0:min(inputs.shape[0],30),0:,0:,0:]
				fix_labels = labels[0:min(inputs.shape[0],30)]
				fix_batch = self.order_batch_by_label(fix_batch,fix_labels)
			for i in range(1):
				log_prob_correct_temp=torch.zeros(1)
				if not self.opts.netopts.customdict["exact"]:
					# Stochastic
					priorcoef= 1
					with torch.autograd.set_detect_anomaly(False):
						loss,\
						output,\
						trys,\
						log_prob_correct_temp,\
						stats= self.likelihood_optimization(inputs,labels,usemin=False,concentration=1)
						(-self.model.get_lrob_model()[0]/128).backward()
						# self.prior_optimization(inputs, labels, coef=1.0, mode='likelihood',concentrate=10)
						# self.renyi_prior_MCMC(inputs,coef=priorcoef)
						if self.opts.netopts.customdict['divgreg']:
							pass
							# self.prior_optimization(inputs, labels,coef=self.opts.netopts.customdict['reg_coef'])
						# loss, output = self.model_stats(inputs,labels)
						outputfull = output
				else:
					trys= 1
					# Exact
					output,logprob,model_prob = self.model(inputs)
					outputfull = output
					output = output.view(-1, self.opts.dataopts.classnum)
					loss = self.opts.optimizeropts.loss(output, labels).mean()
					(loss).backward()
			if batch_n % 10 == -1:
				self.model.print(fix_batch, epoch, batch_n)
			thisNorm = self.block_grad(list(self.model.parameters()))

			self.optimizer.step()
			self.optimizer.zero_grad()
			# self.model.paint_stochastic_graph()

			''' Print Output'''
			#TODO: Print Batch Statistics
			predlab = torch.argmax(output, 1, keepdim=False).squeeze()
			meanoutput = output.logsumexp(dim=0,keepdim=True)
			meanoutput = meanoutput - meanoutput.logsumexp(dim=1,keepdim=True)
			jsd= (output.exp()*output).sum(dim=1).mean() - (meanoutput.exp()*meanoutput).sum()
			run_loss += loss.item()
			run_lprob_correct += log_prob_correct_temp
			accthis = (predlab == labels).sum().item()
			corrects += accthis
			totalsamples += labels.size(0)
			train_acc = corrects/totalsamples
			train_loss = (run_loss/(batch_n+1))
			trytotal += trys
			tryavg = trytotal/(batch_n+1)
			train_avg_prob_correct = run_lprob_correct/(batch_n+1)
			jsdtotal = (jsd + jsdtotal)/2 if batch_n !=0 else jsd
			print("",end='\r')
			print(' '+ prefixprint +':' 
				  'batch: %d '%(batch_n) +
				  ' train_likelihood: %.4f'% train_loss +
			      ' train_posterior: %.4f' % (thisNorm) +
				  ' model_lprob: %f' % (train_avg_prob_correct).item() +
				  ' trials: %.4f' % tryavg+
					'jsd: %.3f' % jsdtotal +
			      ' train accuracy: %.2f'% (train_acc*100),end=" "

			      )
		scalar_dict = self.model.get_scalar_dict()
		self.print(' '+ prefixprint + ':'
		                    'batch: %d ' % (batch_n) +
		      ' train_likelihood: %.4f' % train_loss +
		      ' train_lprob: %.2f' % train_avg_prob_correct +
		      ' train accuracy: %.2f' % (train_acc * 100)+
		           ' trials: %.4f' % tryavg +
		           'jsd: %.3f' % jsdtotal
		           ,end=" "
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
				output= 0
				if self.opts.netopts.customdict['exact']:
					trials =1
				else:
					trials = 10
				for i in range(trials):
					output_temp,logprob,model_prob = self.model(inputs)
					output_temp = output_temp-output_temp.logsumexp(dim=1,keepdim=True)
					if i==0:
						output = output_temp
					else:
						output = LSE_pair(output_temp,output)

				output = (output -math.log(trials))

				output = output.view(-1, self.opts.dataopts.classnum)
				#TODO: Print Batch Statistics
				predlab = torch.argmax(output, 1, keepdim=False).squeeze()
				accthis = (predlab == labels).sum().item()
				val_corrects += accthis
				totalsamples += labels.size(0)
				val_acc = val_corrects/totalsamples
				loss = self.opts.optimizeropts.loss(output,labels).mean()
				val_run_loss += loss.item()
				val_avg_loss = val_run_loss/(batch_n+1)

		self.print(' '+ prefixprint + ':' +
		      ' val_loss: %.4f' % val_avg_loss +
		      ' val_acc: %.2f' % (val_acc * 100))
		resdict = dict(test_acc=val_acc*100,
		               train_acc=train_acc*100,
		               test_loss=val_avg_loss,
		               train_loss=train_loss,
		               train_jsd=jsdtotal,
		               train_avg_try=tryavg)
		resdict.update(scalar_dict)
		return resdict,outputfull

	def run_many_epochs(self,path:str,save_result):
		self.model.to(self.opts.epocheropts.device)
		self.results = ResultStruct(path)
		self.path = path
		self.opts.print(printer=self.print)
		for epoch in range(self.opts.epocheropts.epochnum):
			prefixtext = 'Epoch %d' % epoch
			epochres,outputsample = self.run_epoch(prefixtext,epoch,path)

			self.results.add_epoch_res_dict(epochres,epoch,save_result)
			self.opts.optimizeropts.lr_sched.step()
			#with torch.set_grad_enabled(False):
			#	generated_images = self.model.generate(sample(outputsample,1,1)[0])
			#	imagepath = './GenImages/Images'+ '_epoch_'+ str(epoch+1)+'.bmp'
			#	save_image(generated_images,imagepath,normalize=True,scale_each=True)

		self.model.to(torch.device('cpu'))
		return self.results

	def print(self,string,end='\n'):
		path = self.path
		log_file = open(os.path.join(path,'log.txt'),"a")
		print(str(string),end=end)
		log_file.write(str(string)+end)
		log_file.close()

