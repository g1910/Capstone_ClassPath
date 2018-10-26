import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import ipdb
import pickle
import numpy as np
from tqdm import tqdm

from logger import Logger

from supervisor.sup_policy import SupervisorPolicy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/gauravm/.torch/', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='/home/gauravm/.torch/', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

logger = Logger(os.path.join('./checkpoint',args.exp_name), name='main')

## Resnet pre-trained network

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()

checkpoint = torch.load('res/resnet-18-py3.pth')
net.load_state_dict(checkpoint, strict=False)
net.cuda()

net.eval()

## Supervisor Policy

sup_policy = SupervisorPolicy(256)
sup_policy.cuda()

def train(epoch):
    print('\nEpoch: %d Training' % epoch)
    sup_policy.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1).repeat(1, 10)

        one_hot = torch.zeros((inputs.size(0), 10)).fill_(1).to(device)

        correct_targets = torch.zeros(inputs.size(0), 10).type(
            torch.cuda.FloatTensor)
        correct_targets = correct_targets.scatter_(dim=1, index=targets,
                                                   src=one_hot)
        correct_targets = correct_targets.to(device)

        incorrect_targets_idx = (targets + torch.from_numpy(
            np.random.randint(1, 10, size=(targets.size(0), 1))).to(
            device)) % 10
        incorrect_targets = torch.zeros(inputs.size(0), 10).type(
            torch.cuda.FloatTensor)
        incorrect_targets = incorrect_targets.scatter_(dim=1,
                                                       index=incorrect_targets_idx,
                                                       src=one_hot)
        incorrect_targets = incorrect_targets.to(device)


        pred










