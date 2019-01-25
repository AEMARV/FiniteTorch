
import torch.nn as nn
from  netparsers.staticnets import StaticNet,DenseNet
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
import random
class Epocher(object):
	def __init__(self,opts:allOpts):

		super(Epocher, self).__init__()
		self.opts = opts
		self.trainloader, self.testloader = self.create_data_set(opts) # type:DataLoader
		self.model = self.create_model_module(opts,sample_data=self.trainloader.__iter__().next()[0]) #type:MyModule
		self.optimizer = self.create_optimizer(opts,self.model)
		self.results=None

	def create_data_set(self, opts):

		train_loader, test_loader = self.opts.dataopts.get_loaders(self.opts)

		return train_loader, test_loader

	def reinstantiate_model(self):
		self.model = self.create_model_module(self.opts, sample_data=None)  # type:MyModule
		self.optimizer = self.create_optimizer(self.opts, self.model)
		self.results = None

	def create_model_module(self,opts,sample_data=None) -> StaticNet:
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

	def block_grad(self,paramlist:List,ind=None):
		length = paramlist.__len__()
		if ind is None:
			ind = random.randint(0, length-1)

		for i in range(length):
			if i == ind:
				paramlist[i].grad = paramlist[i].grad
				continue
			paramlist[i].grad = None

	def normalize_grad(self,paramlist:List):
		length = paramlist.__len__()

		for i in range(length):
			g = paramlist[i].grad
			paramlist[i].grad = paramlist[i].grad/ float((math.log(g.shape[1])))

	def backward_reject_stoch(self,input,labels):
		batchsz = input.shape[0]
		accepted = False
		outputall = None
		labelall = []
		index=  0
		runloss= 0
		runoutput= 0
		lprob_currect= 0
		while True:
			index += input.shape[0]
			output,logprob,logprob_model= self.model(input)
			outputsample,logprob_out = sample(output,1,1)
			logprob = logprob.squeeze() + logprob_out.sum(dim=(2,3),keepdim=True).squeeze() #+ minprob.squeeze()
			outputlabelsample = outputsample.max(dim=1,keepdim=True)[1]

			# calc accept/reject masks
			accepted_temp = outputlabelsample.squeeze() == labels
			#to_be_summed = ((accepted^1) * accepted_temp).squeeze().detach()
			to_be_summed = (  accepted_temp).squeeze().detach()
			#accepted = accepted_temp | accepted #type: Tensor

			loss = self.opts.optimizeropts.loss(output.view(-1,self.opts.dataopts.classnum),labels)
			if index/batchsz ==1:
				runloss = (loss).detach()
				runoutput = output.detach()

			lprob_currect += (((-logprob.squeeze() * (to_be_summed.float())) / batchsz).sum()).detach()
			loss = (-logprob.squeeze()*((to_be_summed).float()))/batchsz
			if to_be_summed.float().sum()>0:
				pass
				loss.sum().backward()
			input = self.logical_index(input,to_be_summed^1)
			labels = self.logical_index(labels,to_be_summed^1)

			if to_be_summed.all():
				runloss = (runloss).mean()
				runoutput = (runoutput)
				break
		index = float(index)/float(batchsz)
		return runloss,runoutput,index,lprob_currect

	def backward_mile_stoch(self,input,labels):
		batchsz = input.shape[0]
		accepted = False
		outputall = None
		labelall = []
		index=  0
		runloss= 0
		remain = torch.ones(batchsz).to('cuda:0')/batchsz
		while True:

			output= self.model(input)
			loss = self.opts.optimizeropts.loss(output.view(-1,self.opts.dataopts.classnum),labels)
			labelprob = (-loss.detach()).exp()
			to_be_summed = torch.min(labelprob/batchsz, remain)
			remain = remain - to_be_summed
			runloss = loss.mean()
			loss = loss*to_be_summed

			loss.sum().backward()

			#rint(index)
			if (remain==0).all():
				break
		return runloss,output

	def backward_single_stoch(self,input,labels):
		batchsz = input.shape[0]
		accepted = False
		outputall = None
		labelall = []
		index=  0
		runloss= 0

		output= self.model(input)
		outputlabelsample = sample(output,1,1)[0].max(dim=1,keepdim=True)[1]

		# calc accept/reject masks
		accepted_temp = outputlabelsample.squeeze() == labels
		to_be_summed = ((accepted^1) * accepted_temp).squeeze().detach()
		accepted = accepted_temp | accepted #type: Tensor

		loss = self.opts.optimizeropts.loss(output.view(-1,self.opts.dataopts.classnum),outputlabelsample.squeeze())
		runloss +=self.opts.optimizeropts.loss(output.view(-1,self.opts.dataopts.classnum),labels).mean().detach()
		loss = loss*(2*to_be_summed.float()-1)

		loss.mean().backward()

		#rint(index)

		return runloss,output,1

	def order_batch_by_label(self,batch,labels):
		_,indices = labels.sort(dim=0)
		batch = batch[indices,:,:,:]
		return batch
	def logical_index(self,batch:Tensor,booleanind):
		int_index= booleanind.nonzero().squeeze()
		return batch.index_select(dim=0,index=int_index)

	def run_epoch(self,prefixprint:str,epoch)->dict:
		run_loss = 0.0
		run_lprob_correct=0.0
		corrects = 0
		totalsamples = 0
		self.model.train()
		regloader = iter(self.trainloader)
		# TODO: Train on batches
		for batch_n,data in enumerate(self.trainloader):

			inputs, labels = data
			inputs, labels = inputs.to(self.opts.epocheropts.device),labels.to(self.opts.epocheropts.device)
			if batch_n ==0:
				fix_batch= inputs[0:min(inputs.shape[0],30),0:,0:,0:]
				fix_labels = labels[0:min(inputs.shape[0],30)]
				fix_batch = self.order_batch_by_label(fix_batch,fix_labels)
			#inputs = fix_batch
			#labels = fix_labels
			self.optimizer.zero_grad()
			for i in range(1):
				log_prob_correct_temp=torch.zeros(1)
				trys=1
				if not self.opts.netopts.customdict["exact"]:
					# Stochastic
					loss,output,trys,log_prob_correct_temp = self.backward_reject_stoch(inputs,labels)
				else:

					trys= 1
					# Exact
					output,logprob,model_prob = self.model(inputs)
					#output = output - output.max(dim=1,keepdim=True)[0]
					#sampleoutput,logprob_out = sample(output,1,1)
					#logprob = logprob_out
					#corr = (sampleoutput.max(dim=1)[1].squeeze()== labels).float()
					#loss_err_acc = ((2*corr -1).squeeze()*(logprob.squeeze())).mean()
					#(-loss_err_acc).backward()
					outputfull = output
					output = output.view(-1, self.opts.dataopts.classnum)
					loss = self.opts.optimizeropts.loss(output, labels).mean()
					(loss+ (-model_prob/50000)).backward()
			if self.opts.netopts.customdict['divgreg']:
				while(True):
					data = next(regloader)
					inputs, _ = data
					inputs, _ = inputs.to(self.opts.epocheropts.device), labels.to(self.opts.epocheropts.device)
					uniforminput= inputs
					uniforminput=  (inputs.exponential_()).clamp(definition.epsilon,None)
					uniforminput = uniforminput / uniforminput.sum(dim=1,keepdim=True)
					uniforminput = uniforminput.log()


					output_this,_,_ = self.model(uniforminput)
					output_unif = output_this.exp()*0
					output_unif = output_unif - output_unif.logsumexp(dim=1,keepdim=True)
					divg=(output_unif - output_this)
					divg = divg.min(dim=1,keepdim=True)[0]

					divgsamp = ((divg.exp()) > torch.rand_like(divg)).float()
					if divgsamp.sum()==0:
						continue
					#divg= ((divgsamp * divg).sum(dim=0))/divgsamp.sum()
					divg = divg.mean()/5e4
					break

				(-divg).backward()


				# Loss Components
				#log_prob = self.model.get_log_prob()
				#regularizer = self.model.get_reg_vals()

				#raw_loss = self.opts.optimizeropts.loss(output.view(-1, self.opts.dataopts.classnum), labels).mean()
				#loss= regularizer
				# End Loss Components
				#raw_loss = loss.mean()



			if (batch_n+1) % 1 ==0:
				#paramlist = list(self.model.parameters())
				#self.block_grad(paramlist,ind =paramlist.__len__()- (epoch//1)%paramlist.__len__())
				#self.optimizer.step()
				#self.optimizer.zero_grad()
				here = False
				if here and not self.opts.netopts.customdict['exact']:
					uniforminput = inputs * 0
					uniforminput = uniforminput - uniforminput.logsumexp(dim=1,keepdim=True)
					output_tmp, maxprob, model_prob  = self.model.forward(uniforminput,MAP=True)
					_, logprob_out = sample_map(output_tmp, 1, 1)
					maxprob = maxprob.squeeze() + logprob_out.sum(dim=(2, 3,4), keepdim=True).squeeze()
					(maxprob/50000).mean(dim=0).backward()
				self.optimizer.step()
				self.optimizer.zero_grad()

			''' Print Output'''
			#self.model.print(fix_batch,epoch,batch_n)
			#expected_logprob = log_prob.mean()
			#expected_regularizer = regularizer.mean()
			#TODO: Print Batch Statistics
			predlab = torch.argmax(output, 1, keepdim=False)
			meanoutput = output.logsumexp(dim=0,keepdim=True)
			meanoutput = meanoutput - meanoutput.logsumexp(dim=1,keepdim=True)
			jsd= (output.exp()*output).sum(dim=1).mean() - (meanoutput.exp()*meanoutput).sum()
			run_loss += loss.item()
			run_lprob_correct += log_prob_correct_temp.item()
			accthis = (predlab == labels).sum().item()
			corrects += accthis
			totalsamples += labels.size(0)
			train_acc = corrects/totalsamples
			train_loss = (run_loss/(batch_n+1))
			train_avg_prob_correct = run_lprob_correct/(batch_n+1)
			print(' '+ prefixprint +':' 
				  'batch: %d '%(batch_n) +
				  ' train_loss: %.4f'% train_loss +
				  ' train_lprob: %.2f' % train_avg_prob_correct +
				  ' trials: %.2f' % trys+
					'jsd: %.3f' % jsd +
			      ' train accuracy: %.2f'% (train_acc*100),end="\r"

			      )
		scalar_dict = self.model.get_scalar_dict()
		print(' '+ prefixprint + ':'
		                    'batch: %d ' % (batch_n) +
		      ' train_loss: %.4f' % train_loss +
		      ' train_lprob: %.2f' % train_avg_prob_correct +
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
				output= 0
				if self.opts.netopts.customdict['exact']:
					trials =1
				else:
					trials = 50
				for i in range(trials):
					output_temp,logprob,model_prob = self.model(inputs)
					output = output + output_temp.exp()
				output = (output / trials).log()

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
		return resdict,outputfull

	def run_many_epochs(self,path:str,save_result):
		self.model.to(self.opts.epocheropts.device)
		self.results = ResultStruct(path)
		for epoch in range(self.opts.epocheropts.epochnum):
			prefixtext = 'Epoch %d' % epoch
			epochres,outputsample = self.run_epoch(prefixtext,epoch)

			self.results.add_epoch_res_dict(epochres,epoch,save_result)
			self.opts.optimizeropts.lr_sched.step()
			#with torch.set_grad_enabled(False):
			#	generated_images = self.model.generate(sample(outputsample,1,1)[0])
			#	imagepath = './GenImages/Images'+ '_epoch_'+ str(epoch+1)+'.bmp'
			#	save_image(generated_images,imagepath,normalize=True,scale_each=True)

		self.model.to(torch.device('cpu'))
		return self.results


