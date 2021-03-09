# coding:utf8
import torch.nn as nn
import torch
from models.basic_module import BasicModule
from torch.nn import functional as F

class ResBlock(nn.Module):
    def __init__(self,inc,outc,stride=1,shoutcut = None):
        super(ResBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inc,outc,3,stride,1,bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,3,1,1,bias=False),
            nn.BatchNorm2d(outc),
        )
        self.right = shoutcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right==None else self.right(x)
        out+=residual
        return F.relu(out)

class ResNet(BasicModule):
    def __init__(self,num_class=2):
        super(ResNet, self).__init__()
        self.model_name ="resnet"

        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1=self.create_layer(64,128,3)
        self.layer2 = self.create_layer(128,256,4,stride=2)
        self.layer3 = self.create_layer(256,512,6,stride=2)
        self.layer4 = self.create_layer(512,256,3,stride=2)
        self.fc = nn.Linear(256,num_class)



    def create_layer(self,inc,outc,block_num=4,stride=1):
        shoutcut = nn.Sequential(
            nn.Conv2d(inc,outc,1,stride,bias=False),
            nn.BatchNorm2d(outc)
        )
        layers=[]
        layers.append(ResBlock(inc,outc,stride,shoutcut))
        for i in range(block_num):
            layers.append(ResBlock(outc,outc))
        return nn.Sequential(*layers)


    def forward(self,x):
        x=self.pre(x)
        x= self.layer1(x)
        x= self.layer2(x)
        x= self.layer3(x)
        x= self.layer4(x)

        x = F.avg_pool2d(x,7)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x




