import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as py


class clip_3dcnn(nn.Module):
    def __init__(self):
        super().__init__(clip_3dcnn,self)
        self.cnn1=nn.Conv3d(in_channels=3,out_channels=3,kernel_size=[2,2,2],stride=2)
        self.cnn2=nn.Conv3d(in_channels=3,out_channels=3,kernel_size=[2,2,2],stride=2)
        self.cnn3=nn.Conv3d(in_channels=3,out_channels=3,kernel_size=[2,2,2],stride=2)
        self.pool1=nn.AvgPool3d(kernel_size=[2,2,2],stride=2)
        self.pool2=nn.AvgPool3d(kernel_size=[2,2,2],stride=2)
        self.pool3=nn.AvgPool3d(kernel_size=[2,2,2],stride=2)
        self.liner1=nn.Linear(in_features=64,out_features=64)
    
    def forward(self,image_features):
        x=self.pool1(self.cnn1(image_features))
        x=self.pool1(self.cnn1(x))
        x=self.pool1(self.cnn1(x))
        x=F.relu(self.liner1(x))
        return x