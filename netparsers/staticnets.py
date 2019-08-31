import torch.nn as nn
from optstructs import *
from netparsers.parseutils import *
import torch.tensor
import math
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import networkx.drawing

class stochasticGraph(object):
	def __init__(self):
		pass
	@staticmethod
	def node_id(layer,state):
		return 'H'+str(layer)+'='+str(state)
	@staticmethod
	def edge_container(matlist):
		for lnum,mat in enumerate(matlist):
			for R in range(mat.shape[0]):
				for D in range(mat.shape[1]):
					thisdict= dict(color=mat[R,D].exp().item())
					idD = stochasticGraph.node_id(lnum,D)
					idR = stochasticGraph.node_id(lnum+1, R)
					yield (idD,idR,thisdict)
	@staticmethod
	def node_container(matlist:List):
		matlist = matlist
		for lnum,mat in enumerate(matlist):
			for D in range(mat.shape[1]):
				idD = stochasticGraph.node_id(lnum,D)
				yield (idD,dict(pos=(lnum,D/mat.shape[1])))

		m = matlist[-1]
		for D in range(mat.shape[0]):
			lnum = matlist.__len__()
			idD = stochasticGraph.node_id(lnum,D)
			yield (idD,dict(color='red',pos=(lnum,D/mat.shape[0])))



class StaticNet(MyModule):
	''' A static Module constructed from a model string. The model string specs are parsed with the static functions
	in the class.
	Constructor:

	StaticNet(modelstring, opts)

	modelstring: the model specification string with delimiter '->'
	opts: opts struct.
	'''
	def __init__(self,modelstring,inputchannels,weightinit=None,biasinit=None,sample_data=None):
		super(StaticNet, self).__init__(blockidx=0)
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		self.layerlist = self.parse_model_string(modelstring,inputchannels['chansz'],inputchannels['icnum'])
		for bloacknum,layer in enumerate(self.layerlist):
			if isinstance(layer,nn.Conv2d):
				weightinit(layer.weight.data)
				biasinit(layer.bias.data)

			self.add_module('block'+str(bloacknum),layer)
		self.sampler = Sampler(blockidx=-1)

	def generate(self,y):

		for l in reversed(self.layerlist):
			if isinstance(l, torch.nn.Conv2d):
				#y = y - l.add_bias(y,)
				bias_reshaped = l.bias.view(1,y.shape[1],1,1)
				y = y - bias_reshaped.data
				wnorm = (l.weight.data**2).sum(dim=1,keepdim=True).sum(dim=2,keepdim=True).sum(dim=3,keepdim=True).sqrt()
				w = l.weight.data/wnorm

				weightmat = l.weight.view(3,3).inverse().view(3,3,1,1)
				y = F.conv_transpose2d(y,weightmat)
			elif isinstance(l, torch.nn.Sigmoid):
				y=y

			else:
				y= l.generate(y)


		return y

	def get_stochastic_mats(self):
		stochastic_list = []
		for layer in self.modules():
			if isinstance(layer,BayesFunc):
				m = layer.get_stochastic_mat()
				stochastic_list = stochastic_list + [m]
		return  stochastic_list

	def paint_stochastic_graph(self):
		''' gets a list of stochastic matrices as input or from the object and paints the graph
		The intensity of the edges of the garph represents the transition probability
		'''
		graph = nx.DiGraph()

		m = self.get_stochastic_mats()
		m_first= m[0]
		input_sz = m_first.shape[1]
		graph.add_nodes_from(stochasticGraph.node_container(m))
		graph.add_edges_from(stochasticGraph.edge_container(m))
		edges = graph.edges()
		pos= graph.nodes(data='pos')
		colors = [graph[u][v]['color'] for u, v in edges]
		plt.cla()
		nx.draw_networkx(graph,node_size=10,edge_color=colors,pos=pos,edge_cmap=plt.cm.Reds,edge_vmin=0,edge_vmax=1)
		plt.show(block=False)
		plt.pause(0.000001)


	def accumulate_lprob(self,lp,usemin=False):

		if lp is None: return None
		if usemin:
			# minlp = lp.min(dim=1, keepdim=True)[0]\
			# 	.min(dim=2, keepdim=True)[0]\
			# 	.min(dim=3, keepdim=True)[0]\
			# 	.min(dim=4,keepdim=True)[0].squeeze()
			minlp = -(-lp).logsumexp(dim=(1,2,3,4),keepdim=True).squeeze()
			return minlp
		else:
			sumlp = lp.sum(dim=(1, 2, 3, 4), keepdim=True).squeeze()
			return sumlp
	def accumulate_lprob_pair(self,lp1,lp2,usemin=False):
		if lp1 is None: return lp2
		if lp2 is None: return lp1
		if usemin:
			# minlp = torch.min(lp1,lp2)
			minlp = softmin_pair(lp1,lp2)
			return minlp
		else:
			return  lp1+lp2
	def p_invert(self,state_list,y):
		state_list.reverse()
		sampler_id = 0
		logprob= None
		for i, layer in enumerate(reversed(self.layerlist)):
			if isinstance(layer, Sampler):
				y,logprob_temp =layer.p_invert(y,state_list[sampler_id])
				logprob_temp = self.accumulate_lprob(logprob_temp)
				logprob = self.accumulate_lprob_pair(logprob, logprob_temp)
				sampler_id += 1
			elif isinstance(layer, MyModule):
				y = layer.p_invert(y)
		return logprob

	def jsd(self,xl):
		xl = xl.detach()
		xl = xl- xl.logsumexp(dim=1,keepdim=True)
		log_dim= math.log(xl.shape[1])

		xp = xl.exp()
		mean_ent = -xl*xp
		mean_ent = mean_ent.sum(dim=1,keepdim=True).mean()/log_dim
		xp_mean = xp.mean(dim=(0,2,3),keepdim=True)
		xl_mean = xp_mean.log()
		ent_mean = -xp_mean*xl_mean
		ent_mean = ent_mean.sum(dim=1,keepdim=True).mean()/log_dim
		return ent_mean-mean_ent

	def get_lrob_model(self):
		prior = None
		lrob = 0
		for i, layer in enumerate(self.layerlist):
			if isinstance(layer, BayesFunc):
				temp_lrob, prior = layer.get_lrob_model(prior)
				lrob = lrob + temp_lrob.sum()

		return lrob,None
	def forward(self, x:Tensor,mode='likelihood',usemin=False,concentration=1.0):
		# Max pooling over a (2, 2) window
		logprob = None
		model_prob = 0
		rnd= None
		useminpair= usemin
		stat_dict = dict(jsd=[])
		prior = None
		# rnd = torch.rand(x.shape[0],1,1,1,1,dtype=x.dtype).to(x.device)
		for i,layer in enumerate(self.layerlist):
			if not isinstance(layer,MyModule):
				x = layer(x)
			else:
				if isinstance(layer,Sampler):
					stat_dict['jsd']= stat_dict['jsd'] + [self.jsd(x).detach().item()]
					x,logprob_temp= layer(x, mode=mode, manualrand=rnd,concentration=concentration)
					logprob_temp = self.accumulate_lprob(logprob_temp,usemin=usemin)
					logprob = self.accumulate_lprob_pair(logprob_temp,logprob,usemin=useminpair)

				elif isinstance(layer,BayesFuncI):
					x, logprob_temp = layer(x, mode=mode, manualrand=rnd,concentration=concentration)
					logprob = self.accumulate_lprob_pair(logprob_temp, logprob, usemin=useminpair)
				elif isinstance(layer,BayesFunc):
					x, logprob_temp = layer(x, mode=mode, manualrand=rnd,concentration=concentration)
					# lrobmodel,prior = layer.get_lrob_model(prior)
					# logprob_temp = lrobmodel + (x*0)
					logprob_temp = self.accumulate_lprob(logprob_temp, usemin=usemin)
					logprob = self.accumulate_lprob_pair(logprob_temp, logprob, usemin=useminpair)

				else:
					x = layer(x,isuniform=False,isinput = i==0)

			if hasnan(x):
				raise Exception(str(layer) + 'has nan')
		return x,logprob,stat_dict

	def forward_intersect(self, x:Tensor,mode='likelihood',usemin=False):
		# Max pooling over a (2, 2) window
		logprob = None
		model_prob = 0
		rnd= None
		stat_dict = dict(jsd=[])
		# rnd = torch.rand(x.shape[0],1,1,1,1,dtype=x.dtype).to(x.device)
		for i,layer in enumerate(self.layerlist):
			if not isinstance(layer,MyModule):
				x = layer(x)
			else:
				if isinstance(layer,Sampler):
					stat_dict['jsd']= stat_dict['jsd'] + [self.jsd(x).detach().item()]
					x,logprob= layer(x, mode=mode, manualrand=rnd, logprob_accumulate=logprob)

				elif isinstance(layer,BayesFunc):
					x, logprob = layer.forward_intersect(x,logprob, mode=mode, manualrand=rnd)
				else:
					x = layer(x,isuniform=False,isinput = i==0)

			if hasnan(x):
				raise Exception(str(layer) + 'has nan')
		return x,logprob,stat_dict
	def forward_unif(self, x:Tensor,MAP=False):
		# Max pooling over a (2, 2) window
		logprobin = torch.zeros(1).to(x.device).squeeze()
		logprobout = torch.zeros(1).to(x.device).squeeze()
		for i,layer in enumerate(self.layerlist):
			if not isinstance(layer,MyModule):
				x = layer(x)
				logprobin_tmp = torch.zeros(1).to(x.device)
				logprobout_tmp = torch.zeros(1).to(x.device)
			else:
				if i==0:
					x,logprobin_tmp,logprobout_tmp= layer(x,isuniform=False,isinput=True)
				else:
					x, logprobin_tmp, logprobout_tmp= layer(x,isuniform=True)
			if hasnan(x):
				raise Exception(str(layer) + 'has nan')
			#TODO Changed summing to prob to mining
			logprobin = logprobin + logprobin_tmp
			# logprobin = softmin_pair(logprobin,logprobin_tmp)
			#logprobin = torch.min(logprobin,logprobin_tmp)
			logprobout = logprobout + logprobout_tmp


		return x,logprobin, logprobout

	''' String Parsers'''
	def parse_model_string(self, modelstring:str, in_n_channel,in_icnum):
		layer_list_string = modelstring.split('->')
		layer_list = []
		out_n_channel = in_n_channel
		blockidx_dict= {}
		for blocknum, layer_string in enumerate(layer_list_string, 0):
			layer,out_n_channel,in_icnum= parse_layer_string(layer_string,out_n_channel,in_icnum,blockidx_dict)
			if layer is not None:
				layer_list += [layer]
		return layer_list


class CompositeNet(StaticNet):
	def __init__(self,modelstring,inputchannels,weightinit=None,biasinit=None,sample_data=None,container=True,block_idx_dict=None):
		super(StaticNet,self).__init__(blockidx=0)
		if block_idx_dict is None:
			block_idx_dict={}
		self.InternalModule = None
		self.layer = None
		if type(modelstring) is not list:
			modelstring= modelstring.split('->')


		if len(modelstring)==1:

			out_n_channels=2
			out_ic_num = 3
		else:
			InternalModule = CompositeNet(modelstring[0:-1],inputchannels,container=False,block_idx_dict=block_idx_dict)
			self.add_module('Internal'+ str(self.get_block_num()),InternalModule)
			self.InternalModule= InternalModule
			out_n_channels, out_ic_num = self.InternalModule.out_n_channels,self.InternalModule.out_icnum
		layer, out_n_channels,out_ic_num = parse_layer_string(modelstring[-1],out_n_channels,out_ic_num,block_idx_dict)
		if isinstance(layer,MyModule):
			layer_compact_name = layer.compact_name
		elif layer is None :
			layer_compact_name = "Identity"
		else:
			layer_compact_name = type(layer).__name__
		self.add_module(layer_compact_name + str(self.get_block_num()), layer)
		self.layer = layer
		self.out_n_channels= out_n_channels
		self.out_icnum = out_ic_num


		if container:
			self.prior= Parameter(data=torch.zeros(1,out_n_channels,1,1,out_ic_num),requires_grad=False)
			self.register_parameter('bias',self.prior)

	def get_block_num(self):
		if self.InternalModule is None:
			return 1
		else:
			return self.InternalModule.get_block_num() + 1

	def forward(self, x:Tensor,prior=None):
		#if prior is None:
			#prior = self.prior - self.prior.logsumexp(dim=1,keepdim=True)
		model_prob = 0
		#input_prior = prior if self.layer is None else self.layer.prop_prior(prior)
		h, logprob_h,temp1 = (x,0,0) if self.InternalModule is None else self.InternalModule(x,prior=None)
		y, logprob_y,temp2 = (h,0,0) if self.layer is None else self.layer(h,prior=None)
		return y, logprob_h+logprob_y,temp1 + temp2
	def forward_unif(self,x,prior=None):
		h, logprob_h, temp1 = (x, 0, 0) if self.InternalModule is None else self.InternalModule.forward_unif(x, prior=None)
		if self.InternalModule is None:
			h,logprob_temp = sample(h,1,1)
			logprob_temp *=0
		else:
			h, logprob_temp = sampleunif(h, 1, 1)
		h = (h).log()
		logprob_temp = logprob_temp.sum(dim=(1,2,3,4),keepdim=True)
		y, logprob_y, temp2 = (h, 0, 0) if self.layer is None else self.layer(h, prior=None)
		return y, logprob_h + logprob_y+logprob_temp, temp1 + temp2
	def forward_max(self,x,prior=None):
		h, logprob_h, temp1 = (x, 0, 0) if self.InternalModule is None else self.InternalModule.forward_max(x, prior=None)
		h,logprob_temp = sample_map(h,1,1)
		h = (h).log()
		logprob_temp = logprob_temp.sum(dim=(1,2,3,4),keepdim=True)
		y, logprob_y, temp2 = (h, 0, 0) if self.layer is None else self.layer(h, prior=None)
		return y, logprob_h + logprob_y+logprob_temp, temp1 + temp2



class DenseNet(StaticNet):
	def __init__(self,*args,sample_data=None,**kwargs):
		super(DenseNet, self).__init__(*args,**kwargs)
		# 1 input image channel, 6 output channels, 5x5 square convolution
		# kernel
		convdata = [sample_data]
		self.forward(sample_data.to(device=torch.device('cuda:0')))
		pass
	def forward(self, x:Tensor):
		# Max pooling over a (2, 2) window
		logprob = 0
		x,_ = sample(x,1,1)
		x = x.log()
		convdata=[]
		for layer in self.layerlist:
			if isinstance(layer, KLConv_Base):
				xnew , logprob_temp = layer([x] +convdata)
				xavgd,logprob_temp2 = sample(glavgpool.apply(x)[0],1,1)
				logprob += logprob_temp2
				convdata = [xavgd.log()] + convdata
				x = xnew
			else:
				x,logprob_temp = layer(x)
			logprob += logprob_temp
		return x,logprob

