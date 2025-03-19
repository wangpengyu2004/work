import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
class PositionalEncodeing(nn.Module):
    def __init__(self,frams,dmodel,batch_size):
        super(PositionalEncodeing,self).__init__()
        self.encode=torch.zeros(frams,dmodel)
        pos=torch.range(0,frams,dtype=torch.float32).unsqueeze(1)
        div=torch.exp(torch.range(0,frams,2).float()*(-math.log(10000)/dmodel))
        self.encode[:,0::2]=torch.sin(pos*div)
        self.encode[:,1::2]=torch.cos(pos*div)
        self.encode=self.encode.unsqueeze(0)        
    def forward(self,x):
        return x+self.encode[:,:x.size(2)].detach()
        