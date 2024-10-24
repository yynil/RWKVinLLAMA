import sys
import os
import torch
from model import wind_rwkv7,RWKV7

print(wind_rwkv7)
from argparse import Namespace
args = Namespace(n_embd=512, n_layer=6, dim_att=512)
rwkv7 = RWKV7(args, 0).bfloat16().cuda()
print(rwkv7)

B,T,C = 10,256,512
x = torch.randn(B,T,C).bfloat16().cuda()
print(x)
y = rwkv7(x)
print(y.shape)
print(y)