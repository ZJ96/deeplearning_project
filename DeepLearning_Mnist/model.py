import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,16,(3,3),(1,1),(1,1))
        self.bn1= nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,24,(3,3),(2,2),(1,1))
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3= nn.Conv2d(24,32,(3,3),(2,2),(1,1))
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*7*7,200)
        self.fc2 = nn.Linear(200,10)
        self.relu =nn.ReLU()

    def forward(self,x):
        insize = x.size(0)
        x=self.conv1(x)    #16x28x28
        x=self.bn1(x)
        x=self.relu(x)
        x =self.conv2(x)   #24x14x14
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)    #32x7x7
        x=self.bn3(x)
        x=self.relu(x)
        x= x.view(insize,-1)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=F.log_softmax(x,dim=1)
        return x


def train(model,device,train_loader,optimizer):
    model.train()
    for batch_id,(data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out =model(data)
        loss = F.nll_loss(out,target)
        loss.backward()
        optimizer.step()
        if (batch_id+1)%50==0:
            print("Train :[ {} / {}     Loss:{:.7f} ]"
                  .format(batch_id*len(data),len(train_loader.dataset),loss.item()))

def test(model,device,test_loader):
    model.eval()
    test_loss =0
    correct =0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device),target.to(device)
            out = model(data)
            test_loss += F.nll_loss(out,target,reduction='sum').item()
            pred = out.max(1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        avg_loss = test_loss/len(test_loader.dataset)
        accuracy_precent = correct/len(test_loader.dataset)*100
        print("__Test__ :average loss : {} ,accuracy : {} / {}  =  {:.4f}%".format(avg_loss,correct,len(test_loader.dataset),accuracy_precent))

def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False   #训练集变化不大时使训练加速
    torch.backends.cudnn.enabled = False















