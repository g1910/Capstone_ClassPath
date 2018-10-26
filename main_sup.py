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

from supervisor.sup_net import SupervisorNetwork

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--exp_name', type=str, default='checkpoint')
parser.add_argument('--l1', action='store_true')
parser.add_argument('--compression', default=0.25, type=float)
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

logger = Logger(os.path.join('./checkpoint',args.exp_name), name='main')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()

checkpoint = torch.load('res/resnet-18-py3.pth')
net.load_state_dict(checkpoint, strict=False)
net.cuda()

net.eval()

path_dims = [64, 64, 128, 128, 256, 256, 512, 512]
sup_net = SupervisorNetwork(path_dims)
sup_net.cuda()



#TODO define loss

ranking_loss = nn.MarginRankingLoss(margin=0.1)
mse_loss = nn.MSELoss()
optimizer = optim.Adam(sup_net.parameters(), lr=args.lr)


def path_ranking_loss(pred_path, true_path):
    pred_path_mtrx_1 = pred_path.unsqueeze(2).repeat(1,1,pred_path.size(1))
    pred_path_mtrx_2 = pred_path.unsqueeze(1).repeat(1, pred_path.size(-1), 1)

    true_path_mtrx1 = true_path.unsqueeze(2).repeat(1, 1, true_path.size(1))
    true_path_mtrx2 = true_path.unsqueeze(1).repeat(1, true_path.size(-1), 1)

    target_mtrx = (true_path_mtrx1 > true_path_mtrx2).type(torch.cuda.FloatTensor) \
                  - (true_path_mtrx1 < true_path_mtrx2).type(torch.cuda.FloatTensor)

    # loss = ranking_loss(pred_path_mtrx_1.view(pred_path.size(0), -1),
    #                     pred_path_mtrx_2.view(pred_path.size(0), -1),
    #                     target_mtrx.view(pred_path.size(0), -1))

    # loss = F.margin_ranking_loss(pred_path_mtrx_1, pred_path_mtrx_2, target_mtrx, margin=0.1)

    loss = (pred_path_mtrx_1 - pred_path_mtrx_2)*(-target_mtrx) + 0.1
    loss = torch.clamp(loss, min=0.)
    loss = torch.mean(loss)
    # ipdb.set_trace()

    return loss


# Training
def train(epoch):
    print('\nEpoch: %d Training' % epoch)
    sup_net.train()
    train_ranking_loss = 0
    train_mse_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1).repeat(1, 10)

        one_hot = torch.zeros((inputs.size(0),10)).fill_(1).to(device)

        correct_targets = torch.zeros(inputs.size(0), 10 ).type(torch.cuda.FloatTensor)
        correct_targets = correct_targets.scatter_(dim=1, index=targets, src=one_hot)
        correct_targets = correct_targets.to(device)

        incorrect_targets_idx = (targets + torch.from_numpy(np.random.randint(1,10, size=(targets.size(0), 1))).to(device))%10
        incorrect_targets = torch.zeros(inputs.size(0), 10).type(
            torch.cuda.FloatTensor)
        incorrect_targets = incorrect_targets.scatter_(dim=1, index=incorrect_targets_idx,
                                                   src=one_hot)
        incorrect_targets = incorrect_targets.to(device)

        # class_vec = torch.cat((correct_targets, incorrect_targets), dim=0)
        # img = inputs.repeat(2,1,1,1)

        out, true_correct_paths, true_incorrect_paths = net.forward_with_paths(inputs)

        pred_correct_paths = sup_net(correct_targets, inputs)
        pred_incorrect_paths = sup_net(incorrect_targets, inputs)

        total_path_loss = 0
        total_path_loss_mse = 0
        total_path_loss_ranking = 0
        for i in range(len(path_dims)):
            ipdb.set_trace()
            correct_loss = path_ranking_loss(pred_correct_paths[i], true_correct_paths[i])
            incorrect_loss = path_ranking_loss(pred_incorrect_paths[i], true_incorrect_paths[i])

            correct_loss_mse = mse_loss(pred_correct_paths[i], true_correct_paths[i])
            incorrect_loss_mse = mse_loss(pred_incorrect_paths[i], true_correct_paths[i])

            total_path_loss += (correct_loss + incorrect_loss + correct_loss_mse + incorrect_loss_mse)

            total_path_loss_mse += (correct_loss_mse + incorrect_loss_mse)
            total_path_loss_ranking += (correct_loss + incorrect_loss)

        optimizer.zero_grad()
        total_path_loss.backward()
        optimizer.step()

        # ipdb.set_trace()

        train_mse_loss += total_path_loss.item()
        train_ranking_loss += total_path_loss_ranking.item()

        progress_bar(batch_idx, len(trainloader), 'MSE Loss: %.3f Ranking Loss: %.3f' %
                     (train_mse_loss/(batch_idx+1), train_ranking_loss/(batch_idx+1)))

    logger.scalar_summary('train/mse_loss', train_mse_loss/len(trainloader), epoch)
    logger.scalar_summary('train/ranking_loss', train_ranking_loss / len(trainloader),
                          epoch)

    if not os.path.exists(os.path.join('./checkpoint', args.exp_name, 'models')):
        os.makedirs(os.path.join('./checkpoint', args.exp_name, 'models'))
    torch.save(sup_net.state_dict(), os.path.join('./checkpoint',
                                                  args.exp_name,
                                                  'models',
                                                  'model_{}.pth'.format(epoch)))

def test(epoch):
    print('\nEpoch: %d Testing' % epoch)
    sup_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1).repeat(1, 10)

        one_hot = torch.zeros((inputs.size(0),10)).fill_(1).to(device)

        correct_targets = torch.zeros(inputs.size(0), 10 ).type(torch.cuda.FloatTensor)
        correct_targets = correct_targets.scatter_(dim=1, index=targets, src=one_hot)
        correct_targets = correct_targets.to(device)

        incorrect_targets_idx = (targets + torch.from_numpy(np.random.randint(1,10, size=(targets.size(0), 1))).to(device))%10
        incorrect_targets = torch.zeros(inputs.size(0), 10).type(
            torch.cuda.FloatTensor)
        incorrect_targets = incorrect_targets.scatter_(dim=1, index=incorrect_targets_idx,
                                                   src=one_hot)
        incorrect_targets = incorrect_targets.to(device)

        # class_vec = torch.cat((correct_targets, incorrect_targets), dim=0)
        # img = inputs.repeat(2,1,1,1)

        out, true_correct_paths, true_incorrect_paths = net.forward_with_paths(inputs)

        pred_correct_paths = sup_net(correct_targets, inputs)
        pred_incorrect_paths = sup_net(incorrect_targets, inputs)

        total_path_loss = 0
        for i in range(len(path_dims)):
            correct_loss = path_ranking_loss(pred_correct_paths[i], true_correct_paths[i])
            incorrect_loss = path_ranking_loss(pred_incorrect_paths[i], true_incorrect_paths[i])

            total_path_loss += (correct_loss + incorrect_loss)

        # optimizer.zero_grad()
        # total_path_loss.backward()
        # optimizer.step()

        # ipdb.set_trace()

        test_loss += total_path_loss.item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f' % (test_loss/(batch_idx+1)))

    logger.scalar_summary('test/ranking_loss', test_loss / len(testloader),
                          epoch)

def test_pruned_accuracy(epoch):
    print('\nEpoch: %d Test Pruned Accuracy' % epoch)
    sup_net.eval()

    total = 0
    total_correct = 0
    total_incorrect = 0
    pred_correct = 0
    pred_incorrect = 0
    correct = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs = inputs.to(device)
        vec_targets = targets.to(device)
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

        pred_correct_paths = sup_net(correct_targets, inputs)
        pred_incorrect_paths = sup_net(incorrect_targets, inputs)

        correct_outputs, _ = net.pruned_forward_pass_classpath(inputs, pred_correct_paths,
                                                           [0, 0, 0, 0, 1, 1, 1,
                                                            1], 2)

        incorrect_outputs, _ = net.pruned_forward_pass_classpath(inputs,
                                                               pred_incorrect_paths,
                                                               [0, 0, 0, 0, 1,
                                                                1, 1,
                                                                1], 2)

        total += (correct_outputs.size(0) + incorrect_outputs.size(0))
        _, correct_predicted = correct_outputs.max(1)
        _, incorrect_predicted = incorrect_outputs.max(1)
        batch_correct = correct_predicted.eq(vec_targets).sum().item()
        batch_incorrect = incorrect_outputs.size(0) - incorrect_predicted.eq(vec_targets).sum().item()

        pred_correct += batch_correct
        pred_incorrect += batch_incorrect

        total_correct += correct_outputs.size(0)
        total_incorrect += incorrect_outputs.size(0)

        correct += (batch_correct + batch_incorrect)


    print('accuracy %f', correct/total)

    logger.scalar_summary('train/accuracy', correct/total,
                          epoch)

    logger.scalar_summary('train/accuracy_given_correct_class', pred_correct / total_correct,
                          epoch)

    logger.scalar_summary('train/accuracy_given_incorrect_class',
                          pred_incorrect / total_incorrect,
                          epoch)

    ### testing

    total = 0
    total_correct = 0
    total_incorrect = 0
    pred_correct = 0
    pred_incorrect = 0
    correct = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        inputs = inputs.to(device)
        vec_targets = targets.to(device)
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

        pred_correct_paths = sup_net(correct_targets, inputs)
        pred_incorrect_paths = sup_net(incorrect_targets, inputs)

        correct_outputs, _ = net.pruned_forward_pass_classpath(inputs,
                                                               pred_correct_paths,
                                                               [0, 0, 0, 0, 1,
                                                                1, 1,
                                                                1], 2)

        incorrect_outputs, _ = net.pruned_forward_pass_classpath(inputs,
                                                                 pred_incorrect_paths,
                                                                 [0, 0, 0, 0, 1,
                                                                  1, 1,
                                                                  1], 2)

        total += (correct_outputs.size(0) + incorrect_outputs.size(0))
        _, correct_predicted = correct_outputs.max(1)
        _, incorrect_predicted = incorrect_outputs.max(1)
        batch_correct = correct_predicted.eq(vec_targets).sum().item()
        batch_incorrect = incorrect_outputs.size(0) - incorrect_predicted.eq(
            vec_targets).sum().item()

        pred_correct += batch_correct
        pred_incorrect += batch_incorrect

        total_correct += correct_outputs.size(0)
        total_incorrect += incorrect_outputs.size(0)

        correct += (batch_correct + batch_incorrect)

    print('accuracy %f', correct / total)

    logger.scalar_summary('test/accuracy', correct / total,
                          epoch)

    logger.scalar_summary('test/accuracy_given_correct_class',
                          pred_correct / total_correct,
                          epoch)

    logger.scalar_summary('test/accuracy_given_incorrect_class',
                          pred_incorrect / total_incorrect,
                          epoch)


for epoch in range(start_epoch, start_epoch+200):
    test_pruned_accuracy(epoch)
    test(epoch)
    train(epoch)



