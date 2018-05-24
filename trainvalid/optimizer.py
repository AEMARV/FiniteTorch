from optstructs import *
from torch.nn import Module
from torch.optim import *
import torch
def create_optimizer(opts:allOpts,model:Module):
	optimopts = opts.optimizeropts
	if opts.gpu:
		device = torch.device("cpu")
		model =model.to(device=device)
	optim = globals()[optimopts.type](model.parameters(),
                                      lr = optimopts.lr,
	                               momentum=optimopts.momentum,
	                               weight_decay=optimopts.weight_decay,
	                               dampening=optimopts.dampening,
	                               nesterov=optimopts.nestrov)
	return optim
