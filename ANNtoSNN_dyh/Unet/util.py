# coding=gbk

import os
import time
import pdb

import torch
# import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import shutil

class MedicalDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.samples = self.getScAndFc(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    #传入的index参数作为索引用于选择dataset List中的一个样本，所以就从samples-->sample
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input = sample[0]
        target = sample[1]
        return sample   # {'input':img2, 'target':img2}
    
    def getScAndFc(self, basePath):
        dataInfo = []
        folderList = [os.path.join(basePath, folder) for folder in os.listdir(basePath)]
        # 只取前五个文件夹
        folderList = folderList[:5]
        #print(folderList)
        for folder in folderList:
            #print(os.listdir(folder))
            secondaryPath = [os.path.join(folder, secondaryFolder) for secondaryFolder in os.listdir(folder)]
            #接下来应该是两个for循环分别遍历SC和FC并拼接
            fcPath = [os.path.join(secondaryPath[3], aimFC) for aimFC in os.listdir(secondaryPath[3])]
            scPath = [os.path.join(secondaryPath[1], aimSC) for aimSC in os.listdir(secondaryPath[1])]
            #print(scPath)
            #print(fcPath)
            for fc, sc in zip(fcPath, scPath):
                #tag = fc[-7:-4]
                scArr = np.loadtxt(sc)
                #print(1)
                fcArr = np.loadtxt(fc)
                #print(2)
                #以couple方式进行对应
                dataInfo.append((scArr, fcArr))
                #print((scArr, fcArr))
            #print('----------------------------------------------')
        # print(dataInfo)
        # print(len(dataInfo))
        
        return(dataInfo)

def save_checkpoint(state, is_best, checkpoint_dir):
    print("save checkpoint")
    filename = checkpoint_dir+'/epoch'+str(state['epoch']).zfill(3)+'.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, checkpoint_dir+'/model_best.pth.tar')

def load_checkpoint_by_key(values, checkpoint_dir, keys):
    '''
    the key can be state_dict for both optimizer or model,
    value is the optimizer or model that define outside
    '''
    filename = checkpoint_dir+'/model_best.pth.tar'
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        for i, key in enumerate(keys):
            values[i].load_state_dict(checkpoint[key])
        print("loaded checkpoint from '{}' (epoch: {}, monitor loss: {})".format(filename, \
                epoch, checkpoint['monitor_loss']))
    else:
        raise ValueError('No correct checkpoint')
    return values, epoch


def save_test_result(res, test_dir):
    '''self define function to save results or visualization'''
    print('edit here')
    return
