import torch
from layers.pmaputils import sample
from layers.klmodules import Sampler
a = torch.rand(4,3,2).log().detach()

a= (a- a.logsumexp(dim=1,keepdim=True)).detach()
a.requires_grad=True
total= 0
sampler = Sampler(blockidx=0)
iternum=10000
for i in range(iternum):
	b, logprob = sampler(a,samplelikelihood=True)
	(logprob.sum()/iternum).backward()
	total += b.exp()
total = total /iternum

print(total.mean())