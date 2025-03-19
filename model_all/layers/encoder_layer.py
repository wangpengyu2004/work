import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
#线性层
class VisionEncoder(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(VisionEncoder, self).__init__()
        self.img_fc1 = nn.Linear(inputsize, outputsize)
        #self.img_fc2 = nn.Linear(inputsize, outputsize)
        #self.layernorm1=nn.LayerNorm(outputsize)

    def forward(self, input_img):
        x = self.img_fc1(input_img)
        #x=self.layernorm1(x)
        x=F.relu(x)
        #x = self.img_fc2(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(TextEncoder, self).__init__()
        self.img_fc1 = nn.Linear(inputsize, outputsize)
        #self.img_fc2 = nn.Linear(inputsize, outputsize)
        #self.layernorm2=nn.LayerNorm(outputsize)
    def forward(self, input_text):
        x = self.img_fc1(input_text)
        #x=self.layernorm2(x)
        x=F.relu(x)
        #x = self.img_fc2(x)
        return x