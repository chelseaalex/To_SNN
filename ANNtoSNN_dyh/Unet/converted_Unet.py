import sys
sys.path.append('../../..')
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib
matplotlib.use('agg')
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import os
from braincog.utils import setup_seed
from braincog.datasets.datasets import get_cifar10_data
from braincog.base.conversion import Convertor
from examples.Perception_and_Learning.Conversion.burst_conversion.CIFAR10_VGG16 import VGG16

import argparse
from model import *  # UNet
from util import *



parser = argparse.ArgumentParser(description='Conversion')
parser.add_argument('--T', default=64, type=int, help='simulation time')
parser.add_argument('--p', default=0.99, type=float, help='percentile for data normalization. 0-1')
parser.add_argument('--gamma', default=5, type=int, help='burst spike and max spikes IF can emit')
parser.add_argument('--channelnorm', default=False, type=bool, help='use channel norm')
parser.add_argument('--lipool', default=True, type=bool, help='LIPooling')
parser.add_argument('--smode', default=True, type=bool, help='replace ReLU to IF')
parser.add_argument('--soft_mode', default=True, type=bool, help='soft reset or not')
parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--cuda', default=True, type=bool, help='use cuda.')
parser.add_argument('--model_name', default='vgg16', type=str, help='model name. vgg16 or resnet20')
parser.add_argument('--merge', default=True, type=bool, help='merge conv and bn')
parser.add_argument('--train_batch', default=100, type=int, help='batch size for get max')
parser.add_argument('--batch_num', default=1, type=int, help='number of train batch')
parser.add_argument('--spicalib', default=0, type=int, help='allowance for spicalib')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for testing')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--in_num_ch', dest='in_num_ch', type=int, default=1)
parser.add_argument('--out_num_ch', dest='out_num_ch', type=int, default=1)
parser.add_argument('--input_size', dest='input_size', type=int, default=90, help='resize input image size')
args = parser.parse_args()

def validate(net, valDataLoader, epoch, global_iter, device):
    net.eval()
    loss_G_all = 0
    for iter, sample in enumerate(valDataLoader):
        real_A = sample[0]
        real_A = np.expand_dims(real_A, axis=3) # 因为导入的是二位数组，所以为了适配这个原本处理图像的网络，需要升一维
        real_A = torch.tensor(real_A).permute(0,3,1,2).to(device)
        real_A = real_A.float()
        
        real_B = sample[1]
        real_B = np.expand_dims(real_B, axis=3)
        real_B= torch.tensor(real_B).permute(0,3,1,2).to(device)
        #print(real_B)
        real_B = real_B.float()
        fake_B = net(real_A)

        loss_G_all += F.l1_loss(fake_B, real_B).item()  # need to accumulate using float not tensor, else the memory will be explode
    loss_G_mean = loss_G_all / (iter + 1)
    print('Validation: Epoch: [%2d], L1_loss: %.8f' % (epoch, loss_G_mean))
    
def evaluate_snn(test_iter, snn, device=None, duration=50):
    device=0
    accs = []
    snn.eval()

    for ind, (test_x, test_y) in tqdm(enumerate(test_iter)):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        n = test_y.shape[0]
        out = 0
        with torch.no_grad():
            snn.reset()
            acc = []
            # for t in tqdm(range(duration)):
            for t in range(duration):
                out += snn(test_x)
                result = torch.max(out, 1).indices
                result = result.to(device)
                acc_sum = (result == test_y).float().sum().item()
                acc.append(acc_sum / n)

        accs.append(np.array(acc))
    accs = np.array(accs).mean(axis=0)

    i, show_step = 1, []
    while 2 ** i <= duration:
        show_step.append(2 ** i - 1)
        i = i + 1

    for iii in show_step:
        print("timestep", str(iii).zfill(3) + ':', accs[iii])
    print("best acc: ", max(accs))


if __name__ == '__main__':
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    #setup_seed(seed=args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:%s" % args.device) if args.cuda else 'cpu'

    trainData = MedicalDataset('./ADNI Dataset')
    trainDataLoader = DataLoader(trainData, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valData = MedicalDataset('./ADNI Dataset')
    valDataLoader = DataLoader(valData, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('built data loader')


    net = UNet(in_num_ch=args.in_num_ch, out_num_ch=args.out_num_ch, first_num_ch=64, input_size=args.input_size,
                    output_activation='softplus').to(device)
    net.load_state_dict(torch.load(".\CIFAR10_UNET.pth", map_location=device))

    net.eval()
    net = net.to(device)
    
    converter = Convertor(dataloader=trainDataLoader,
                          device=device,
                          p=args.p,
                          channelnorm=args.channelnorm,
                          lipool=args.lipool,
                          gamma=args.gamma,
                          soft_mode=args.soft_mode,
                          merge=args.merge,
                          batch_num=args.batch_num,
                          spicalib=args.spicalib
                          )
    snn = converter(net)
    print('convert Successfully')

    #evaluate_snn(test_iter, snn, device, duration=args.T)

