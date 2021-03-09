import torch
import torch.nn as nn
import torchvision

from model import train,test,ConvNet,set_seed
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


#定义超参数
batch_size = 300
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

model = ConvNet().to(device)

'''
#绘制模型结构图
# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
# create grid of images
img_grid = torchvision.utils.make_grid(images)
# write to tensorboard
writer = SummaryWriter('./tensorboard')
writer.add_image('four_fashion_mnist_images', img_grid)
writer.add_graph(model, images)
writer.close()
'''

if __name__ =="__main__":
    set_seed(0)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, epochs + 1):
        print("epoch {} :".format(epoch))
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
    torch.save(model.state_dict(),'./model_mnist.pth')

