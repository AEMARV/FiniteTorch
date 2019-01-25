import torch
import layers.klfunctions as F
def randomize_log_prob(shape):
	y = torch.rand(shape).to('cuda:0').log()
	y = y - F.LogSumExp.apply(y,1)
	y = y.detach()
	return y