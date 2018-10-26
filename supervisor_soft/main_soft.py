import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
sys.path.append('/home/gauravm/remote/cifar_class_priors')
from models import ResNet18
from utils import progress_bar

import ipdb
import pickle
import numpy as np
from tqdm import tqdm

from tensorboard_logger import Logger

from supervisor.sup_class import SupervisorQuery

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--exp_name', type=str, default='checkpoint')
parser.add_argument('--l1', action='store_true')
parser.add_argument('--compression', default=0.25, type=float)
parser.add_argument('--log_after_steps', default=100, type=int)
args = parser.parse_args()

start_epoch = 0

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

logger = Logger(os.path.join('./checkpoint',args.exp_name, 'logs'))

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()

checkpoint = torch.load('res/resnet-18-py3.pth')
net.load_state_dict(checkpoint, strict=False)
net.cuda()
net.eval()

path_dim = 128

sup_net = SupervisorQuery(path_dim)
sup_net.cuda()

bce = nn.BCELoss()
smooth_l1 = nn.SmoothL1Loss()
mse = nn.MSELoss()

optimizer = optim.Adam(sup_net.parameters(), lr=args.lr)


def estimate_losses(pred, random_query, binary_target, sup_net):
    query_pred = torch.gather(pred, 1, random_query.view(-1, 1)).squeeze(1)
    # ipdb.set_trace()

    _, class_pred = torch.max(pred, dim=1)
    binary_pred = class_pred.eq(random_query).type(torch.cuda.LongTensor)
    correct = binary_target.eq(binary_pred).sum()
    accuracy = correct.type(torch.cuda.FloatTensor)/binary_target.size(0)

    bce_loss = bce(query_pred, binary_target.type(torch.cuda.FloatTensor))

    one_hot = torch.zeros((10, 10)).fill_(1).to(device)
    s_one_hot = torch.zeros(10, 10).type(
        torch.cuda.FloatTensor)
    s_queries = torch.from_numpy(np.array(list(range(10)))).to(device)
    s_one_hot = s_one_hot.scatter_(dim=1, index=s_queries.view(-1, 1), src=one_hot)

    s_vectors = sup_net(s_one_hot)

    sparsity_loss = smooth_l1(s_vectors, torch.zeros_like(s_vectors).to(device))

    orth_loss = torch.from_numpy(np.float32([0.])).to(device)

    for i in range(10):
        for j in range(i,10):
            orth_loss = orth_loss + torch.dot(s_vectors[i], s_vectors[j])

    saturation_target = s_vectors.detach()>0.5
    saturation_loss = mse(s_vectors, saturation_target.type(torch.cuda.FloatTensor))

    orth_loss = orth_loss/45

    total_loss = bce_loss + sparsity_loss + orth_loss

    losses = {
        'total_loss': total_loss,
        'bce_loss' : bce_loss,
        'smooth_l1_loss': sparsity_loss,
        'orthogonality loss': orth_loss,
        'saturation_loss': saturation_loss,
        'accuracy': accuracy

    }

    return losses

def log_vals(logger, val_dict, step, tag='train'):
    print('Training Step: {} '.format(step),)
    for name in val_dict.keys():
        val = val_dict[name].item()
        logger.log_scalar(tag='{}/{}'.format(tag, name), value=val, step=step)
        print('{}: {} '.format(name, val),)
    print()


def train(epoch, global_step=0):
    print('\nEpoch: %d Training' % epoch)
    sup_net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        one_hot = torch.zeros((inputs.size(0), 10)).fill_(1).to(device)

        random_query = torch.from_numpy(np.random.randint(0, 10, size=(targets.size(0)))).to(device)
        random_one_hot = torch.zeros(inputs.size(0), 10).type(
            torch.cuda.FloatTensor)

        random_one_hot = random_one_hot.scatter_(dim=1, index=random_query.view(-1, 1), src=one_hot)
        random_one_hot = random_one_hot.to(device)
        binary_target = targets.eq(random_query).type(torch.cuda.LongTensor)

        imp_vector = sup_net(random_one_hot)
        out = net.forward_check(inputs, imp_vector)
        out_softmax = F.softmax(out, dim=1)

        losses = estimate_losses(out_softmax, random_query, binary_target, sup_net)

        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        if global_step % args.log_after_steps == 0:
            log_vals(logger, losses, global_step, 'train')

        global_step += 1

    return global_step

global_step = 0
for epoch in range(start_epoch, start_epoch+200):
    global_step = train(epoch, global_step=global_step)


