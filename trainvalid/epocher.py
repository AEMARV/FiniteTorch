from torch.nn import Module
import torch.nn as nn

from torch.utils.data import DataLoader
from optstructs import EpocherOpts
from optstructs import allOpts
from torch.optim import Optimizer
from resultutils.resultstructs import *
import torch
class Epocher(object):
	def __init__(self,model:Module,
				 optimizer:Optimizer,
				 trainloader:DataLoader,
				 testloader:DataLoader,
				 opts:EpocherOpts,
	             optsAll:allOpts,
	             results=None):

		super(Epocher, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.trainloader= trainloader
		self.testloader = testloader
		self.optimizer = optimizer
		self.opts = opts
		self.optall = optsAll
		if results is None:
			self.results = ResultStruct(self.optall)
		else:
			self.results = results

	def run_epoch(self,prefixprint:str)->dict:
		run_loss = 0.0
		totalbatches = 100
		corrects = 0
		totalsamples = 0
		# TODO: Train on batches
		for batch_n,data in enumerate(self.trainloader):
			inputs, labels = data
			inputs, labels = inputs.to(self.opts.device),labels.to(self.opts.device)
			self.optimizer.zero_grad()
			output = self.model(inputs)
			output = output.view(-1,self.opts.classnum)
			loss = self.opts.loss(output,labels)
			loss.backward()
			self.optimizer.step()

			#TODO: Print Batch Statistics
			run_loss += loss.item()
			predlab = torch.argmax(output, 1, keepdim=False)
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
		print(' '+ prefixprint + ':'
		                    'batch: %d ' % (batch_n) +
		      ' train_loss: %.4f' % train_loss +
		      ' train accuracy: %.2f' % (train_acc * 100), end=" "
		      )

		# TODO: Evaluate Test
		val_run_loss = 0
		totalsamples = 0
		val_corrects = 0
		for batch_n,data in enumerate(self.testloader):
			with torch.set_grad_enabled(False):
				inputs, labels = data
				inputs, labels = inputs.to(self.opts.device),labels.to(self.opts.device)
				self.optimizer.zero_grad()
				output = self.model(inputs)
				output = output.view(-1, self.opts.classnum)
				#TODO: Print Batch Statistics
				predlab = torch.argmax(output, 1, keepdim=False)
				accthis = (predlab == labels).sum().item()
				val_corrects += accthis
				totalsamples += labels.size(0)
				val_acc = val_corrects/totalsamples
				loss = self.opts.loss(output,labels)
				val_run_loss += loss.item()
				val_avg_loss = run_loss/(batch_n+1)

		print(' '+ prefixprint + ':' +
		      ' val_loss: %.4f' % val_avg_loss +
		      ' val_acc: %.2f' % (val_acc * 100))
		resdict = dict(val_acc=val_acc*100,train_acc=train_acc*100,val_loss=val_avg_loss,train_loss=train_loss)
		return resdict


	def run_many_epochs(self):
		self.model.to(self.opts.device)
		for epoch in range(self.opts.epochnum):
			prefixtext = 'Epoch %d' % epoch
			epochres = self.run_epoch(prefixtext)
			self.results.add_epoch_res_dict(epochres)
			self.results.draw()


