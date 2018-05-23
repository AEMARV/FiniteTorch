from optstructs import *
from torch.nn import Module
from torch.optim import *
import torch
def create_optimizer(opts:allOpts,model:Module):
	optimopts = opts.optimizeropts
	if opts.gpu:
		device = torch.device("cuda:0")
		model.to(device=device)
	optim = locals()[optimopts.type](model.parameters(),
	                               lr = optimopts.lr,
	                               momentum=optimopts.momentum,
	                               weight_decay=optimopts.weight_decay,
	                               dampening=optimopts.dampening,
	                               nestrov=optimopts.nestrov)
	return optim
