import torch
import layers.klfunctions as F
import mdconv
from cuda.mdconv.test_help_funcs import *

input = randomize_log_prob((100,64,32,32))
input = (input - F.LogSumExp.apply(input,1)).detach()
input.requires_grad_(True)
filt = randomize_log_prob((20,5,3,3))
filt11 = randomize_log_prob((1,64,1,1))
filt11.requires_grad_(True)

myout = mdconv.forward(input,filt11)
myouti = mdconv.iforward(input,filt11)
outtorch = -(input - filt11) #type:torch.Tensor
outtorchi = (input - filt11) #type:torch.Tensor
outtorch = outtorch.min(dim=1,keepdim=True)[0]
outtorchi = outtorchi.min(dim=1,keepdim=True)[0]

outtorch.sum().backward()
dzdintorch = input.grad
dzdfilttorch= filt11.grad
dzdmyin,dzdmyfilt = mdconv.backward(myout*0 + 1, input,filt11 )



z = (outtorch- myout).abs().sum()
z2 = (outtorchi- myouti).abs().sum()
diff_dzdfilt= (dzdfilttorch - dzdmyfilt).abs().sum()
diff_dzdinput = (dzdintorch - dzdmyin).abs().sum()
print(z.item())
print(z2.item())
print(diff_dzdfilt.item(), diff_dzdinput.item())


