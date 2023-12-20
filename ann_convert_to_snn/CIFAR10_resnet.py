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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'D:/Desktop/代码库/Brain-Cog-main/examples/Perception_and_Learning/Conversion/burst_conversion/data/'
from densenet import DenseNet
from resnet import resnet34


def get_cifar10_loader(batch_size, train_batch=None, num_workers=4, conversion=False, distributed=False):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                            CIFAR10Policy(),
                                            transforms.ToTensor(),
                                            Cutout(n_holes=1, length=16),
                                            normalize])
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    train_batch = batch_size if train_batch is None else train_batch
    cifar10_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_test if conversion else transform_train)
    cifar10_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

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
            torch.save(net.state_dict(), './CIFAR10_res.pth')


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

    lr, num_epochs = 0.01, 300
    net = resnet34()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=num_epochs)
    train(net, train_iter, test_iter, optimizer, scheduler, device, num_epochs, losstype='crossentropy')

    net.load_state_dict(torch.load("./CIFAR10_res.pth", map_location=device))
    net = net.to(device)
    acc = evaluate_accuracy(test_iter, net, device)
    print(acc)
