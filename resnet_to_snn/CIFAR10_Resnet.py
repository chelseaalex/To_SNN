import sys
sys.path.append('../../..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time
from braincog.utils import setup_seed
from braincog.datasets.datasets import get_cifar10_data
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/data/datasets'

class ResidualBlock(nn.Module):
    expansion=1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer blocks
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        # Output layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_cifar10_loader(batch_size, train_batch=None, num_workers=4, conversion=False, distributed=False):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.5),
                                            CIFAR10Policy(),
                                            transforms.ToTensor(),
                                            normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    train_batch = batch_size if train_batch is None else train_batch
    cifar10_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=transform_test if conversion else transform_train)
    cifar10_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=False, transform=transform_test)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(cifar10_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(cifar10_test, shuffle=False, drop_last=True)
        train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=train_batch, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
        test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,  pin_memory=True, sampler=val_sampler)
    else:
        train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=train_batch, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_iter, test_iter


def train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='mse'):
    best = 0
    net = net.to(device)
    print("training on ", device)
    if losstype == 'mse':
       loss = torch.nn.MSELoss()
    else:
        loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    losses = []

    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        losss = []
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            label = y
            if losstype == 'mse':
                label = F.one_hot(y, 10).float()
            l = loss(y_hat, label)
            losss.append(l.cpu().item())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        scheduler.step()
        test_acc = evaluate_accuracy(test_iter, net)
        losses.append(np.mean(losss))
        print('epoch %d, lr %.6f, loss %.6f, train acc %.6f, test acc %.6f, time %.1f sec'
              % (epoch + 1, learning_rate, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

        if test_acc > best:
            best = test_acc
            torch.save(net.state_dict(), './CIFAR10_Resnet.pth')


def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]

            if only_onebatch: break
    return acc_sum / n


if __name__ == '__main__':
    setup_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = 128
    train_iter, test_iter, _, _ = get_cifar10_data(batch_size)
    # train_iter, test_iter = get_cifar10_loader(batch_size)
    print('dataloader finished')

    lr, num_epochs = 0.05, 300
    net = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=10)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='crossentropy')

    net.load_state_dict(torch.load("./CIFAR10_Resnet.pth", map_location=device))
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)