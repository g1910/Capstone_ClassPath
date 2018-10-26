'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--exp_name', type=str, default='checkpoint')
parser.add_argument('--l1', action='store_true')
parser.add_argument('--compression', default=0.25, type=float)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('Using l1: {}'.format(args.l1))
print('Using compression: {}'.format(args.compression))

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

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()

checkpoint = torch.load('res/resnet-18-py3.pth')
net.load_state_dict(checkpoint, strict=False)
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.class_prior_modules.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

avg_path_all_classes = pickle.load(open('res/avg_path_all_classes.pkl', 'rb'), encoding='latin1')


def get_compressed_priors(avg_path, c=0.5):
    for i in range(len(avg_path)):
        for j in range(len(avg_path[i])):
            sort_idx = np.argsort(avg_path[i][j])
            drop_idx = np.int32(len(avg_path[i][j]) * c)
            sort_idx_to_drop = sort_idx[:drop_idx]
            avg_path[i][j][sort_idx_to_drop] = 0

    return avg_path


switch_vector = [0,0,0,0,0,0,1,1]
compression = args.compression
class_prior = get_compressed_priors(avg_path_all_classes, compression)



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_l1_reg = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # ipdb.set_trace()
        optimizer.zero_grad()

        batch_loss = 0.
        batch_correct = 0
        batch_l1_reg = 0.

        for j in range(10):
            batch_class_prior = []
            for i in range(8):
                x = np.repeat(np.expand_dims(np.array(class_prior[j][i]),
                                   axis=0), inputs.shape[0], 0)
                batch_class_prior.append(torch.from_numpy(x).type(torch.cuda.FloatTensor).to(device))

            binary_targets = (targets==j).type(torch.cuda.LongTensor)

            outputs, l1_reg = net(inputs, batch_class_prior, switch_vector, reg=True)
            # ipdb.set_trace()

            n_to_binary = torch.zeros(outputs.size(0), outputs.size(1), 2)
            n_to_binary[:,:,0] = 1
            n_to_binary[range(outputs.size(0)), targets, 0] = 0
            n_to_binary[range(outputs.size(0)), targets, 1] = 1

            outputs_binary = torch.matmul(outputs.unsqueeze(1), n_to_binary.cuda()).squeeze(1)
            # ipdb.set_trace()
            loss = criterion(outputs_binary, binary_targets)
            if args.l1:
                loss = loss + 1e-4*l1_reg
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            batch_l1_reg += l1_reg.item() * 1e-4
            _, predicted = outputs_binary.max(1)
            batch_correct += predicted.eq(binary_targets).sum().item()

        # train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        train_l1_reg += batch_l1_reg/10
        train_loss += batch_loss/10
        total += targets.size(0)
        correct += batch_correct/10

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), train_l1_reg/(batch_idx+1),
               100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    test_l1_reg = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            batch_loss = 0.
            batch_correct = 0
            batch_l1_reg = 0.

            for j in range(10):
                batch_class_prior = []
                for i in range(8):
                    x = np.repeat(np.expand_dims(np.array(class_prior[j][i]),
                                                 axis=0), inputs.shape[0], 0)
                    batch_class_prior.append(
                        torch.from_numpy(x).type(torch.cuda.FloatTensor).to(
                            device))

                binary_targets = (targets == j).type(torch.cuda.LongTensor)

                outputs, l1_reg = net(inputs, batch_class_prior, switch_vector, reg=True)
                # ipdb.set_trace()

                n_to_binary = torch.zeros(outputs.size(0), outputs.size(1), 2)
                n_to_binary[:, :, 0] = 1
                n_to_binary[range(outputs.size(0)), targets, 0] = 0
                n_to_binary[range(outputs.size(0)), targets, 1] = 1

                outputs_binary = torch.matmul(outputs.unsqueeze(1),
                                              n_to_binary.cuda()).squeeze(1)
                # ipdb.set_trace()
                loss = criterion(outputs_binary, binary_targets)
                if args.l1:
                    loss = loss + 1e-4 * l1_reg

                batch_loss += loss.item()
                batch_l1_reg += l1_reg.item()
                _, predicted = outputs_binary.max(1)
                batch_correct += predicted.eq(binary_targets).sum().item()



            # outputs = net(inputs)
            # loss = criterion(outputs, targets)
            #
            # test_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            test_l1_reg += batch_l1_reg / 10
            test_loss += batch_loss / 10
            total += targets.size(0)
            correct += batch_correct / 10

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), test_l1_reg/(batch_idx+1),
                   100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pth'.format(args.exp_name))
        best_acc = acc



for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)


