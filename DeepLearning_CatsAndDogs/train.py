# coding =utf-8
import torch
import os
from data_local.dataset import DogAndCat
from models.resnet import ResNet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from opt import OPT
from models import resnext
from models.pretrainedmodel import get_resnet_50, get_resnext_50_32x4d, get_wide_resnet50_2

import numpy as np
import random


def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 并行gpu
    torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速
    torch.backends.cudnn.enabled = True


def train(**kwargs):
    opt = OPT()
    train_data = DogAndCat(opt.train_data_root, train=True, test=False)
    train_dataLoader = DataLoader(train_data, opt.batch, shuffle=True, num_workers=opt.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()

    all_loss = 0.0
    for batch_id, (data, target) in enumerate(train_dataLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        print(out.shape)
        loss = criterion(out, target)
        all_loss += loss
        loss.backward()
        optimizer.step()
        if (batch_id) % opt.train_print_num == opt.train_print_num - 1:
            print("Train :[ {} / {}   Loss:{:.8f} ]"
                  .format(batch_id * len(data), len(train_dataLoader.dataset), all_loss / opt.train_print_num))
            all_loss = 0.0


def test(**kwargs):
    opt = OPT()
    model.eval()
    correct = 0
    test_data = DogAndCat(opt.test_data_root, train=False, test=True)
    test_dataLoader = DataLoader(test_data, opt.batch, shuffle=True, num_workers=opt.num_workers)

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_dataLoader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()

        accu = float(correct.item() / len(test_dataLoader.dataset))
        message = "__Test__ accuracy:  epoch_" + str(kwargs["epoch"]) + "_:    " + str(correct.item()) + " / " + str(
            len(test_dataLoader.dataset)) + "--" * 4 + str(accu)
        print(message)
        # print("__Test__ accuracy : {} / {}---------{}".format(correct,len(test_dataLoader.dataset),accu))
        log_file_name = opt.log_file_name
        with open(log_file_name, 'a') as f:  # 'a'表示append,即在原来文件内容后继续写数据（不清楚原有数据）
            f.write(str(message) + "\n")


if __name__ == "__main__":
    opt = OPT()
    # set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = get_wide_resnet50_2(num_classes=2, pretrained=False).to(device)
    model = ResNet().to(device)
    # model.load_state_dict(torch.load("checkpoint/resnext_50_32x4d_epoch_46.pth"))

    for epoch in range(1, opt.epoch + 1):
        print("\nepoch {} :".format(epoch))
        train()
        test(epoch=epoch)
        if not os.path.exists("./checkpoint"):
            os.mkdir("./checkpoint")
        if epoch % 1 == 0:
            save_path = "./checkpoint/wide_resnet50_2_epoch_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), save_path)











