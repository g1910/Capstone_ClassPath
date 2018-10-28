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
sys.path.append('.')
from models import ResNet18
from utils import progress_bar

import ipdb
import pickle
import numpy as np
from tqdm import tqdm

from tensorboard_logger import Logger

from supervisor.sup_net import SupervisorNetwork

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--exp_name', type=str, default='checkpoint')
parser.add_argument('--l1', action='store_true')
parser.add_argument('--compression', default=0.25, type=float)
parser.add_argument('--log_after_steps', default=100, type=int)
parser.add_argument('--lambda_bce', default=1, type=float)
parser.add_argument('--lambda_ortho', default=1, type=float)
parser.add_argument('--lambda_quant', default=1, type=float)
parser.add_argument('--lambda_l1', default=1, type=float)
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

with open(os.path.join('./checkpoint', args.exp_name, 'args.pkl'), "wb") as f:
    pickle.dump(args, f)

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()

checkpoint = torch.load('res/resnet-18-py3.pth')
net.load_state_dict(checkpoint, strict=False)
net.cuda()
net.eval()

path_dims = [64, 64, 128, 128, 256, 256, 512, 512]
switch_vec = [0, 0, 0, 0, 1, 1, 1, 1]
sup_net = SupervisorNetwork(path_dims)

models_dir = os.path.join('./checkpoint',args.exp_name, 'models')
if os.path.exists(models_dir):
    load_epoch = 0
    for file in os.listdir(models_dir):
        if file.startswith("sup_net_epoch_"):
            load_epoch = max(load_epoch, int(file.split('_')[3]))
    if load_epoch > 0:
        load_filename = 'sup_net_epoch_{}.pth'.format(load_epoch)
        print('Loading model {}'.format(load_filename))
        load_path = os.path.join(models_dir, load_filename)
        sup_net.load_state_dict(torch.load(load_path))


sup_net.cuda()

bce = nn.BCELoss()
smooth_l1 = nn.L1Loss()
mse = nn.MSELoss()

optimizer = optim.Adam(sup_net.parameters(), lr=args.lr)


def estimate_metrics(pred, random_query, binary_target, sup_net, switch_vec):
    query_pred = torch.gather(pred, 1, random_query.view(-1, 1)).squeeze(1)

    num_s = torch.tensor(np.sum(switch_vec).item(), dtype=torch.float32).to(device)
    # ipdb.set_trace()

    _, class_pred = torch.max(pred, dim=1)
    binary_pred = class_pred.eq(random_query).type(torch.cuda.LongTensor)
    correct = binary_target.eq(binary_pred).sum()
    accuracy = correct.type(torch.cuda.FloatTensor)/binary_target.size(0)

    bce_loss = bce(query_pred, binary_target.type(torch.cuda.FloatTensor))

    metrics = {}
    s_hist = {}

    metrics['accuracy'] = accuracy
    metrics['bce_loss'] = bce_loss * args.lambda_bce
    metrics['smooth_l1_loss_total'] = torch.from_numpy(np.float32([0.])).to(device)
    metrics['orthogonality_loss_total'] = torch.from_numpy(np.float32([0.])).to(device)
    metrics['quantization_loss_total'] = torch.from_numpy(np.float32([0.])).to(device)
    metrics['total_loss'] = torch.from_numpy(np.float32([0.])).to(device)
    # ipdb.set_trace()
    metrics['total_loss'] = metrics['total_loss'] + metrics['bce_loss']

    one_hot = torch.zeros((10, 10)).fill_(1).to(device)
    s_one_hot = torch.zeros(10, 10).type(
        torch.cuda.FloatTensor)
    s_queries = torch.from_numpy(np.array(list(range(10)))).to(device)
    s_one_hot = s_one_hot.scatter_(dim=1, index=s_queries.view(-1, 1), src=one_hot)

    s_vectors_all = sup_net(s_one_hot)

    for k in range(len(switch_vec)):
        if switch_vec[k]:
            s_vectors = s_vectors_all[k]
            for i in range(10):
                s_hist['s_layer_{}_class_{}'.format(k, i)] = s_vectors[i].cpu().data.numpy()

            sparsity_loss = smooth_l1(s_vectors, torch.zeros_like(s_vectors).to(device))

            orth_loss = torch.from_numpy(np.float32([0.])).to(device)

            for i in range(10):
                for j in range(i,10):
                    orth_loss = orth_loss + torch.dot(s_vectors[i], s_vectors[j])

            quantization_target = s_vectors.detach()>0.5
            quantization_loss = mse(s_vectors, quantization_target.type(torch.cuda.FloatTensor))

            orth_loss = orth_loss/45

            # ipdb.set_trace()
            metrics['smooth_l1_loss_{}'.format(k)] = sparsity_loss * args.lambda_l1
            metrics['smooth_l1_loss_total'] = metrics['smooth_l1_loss_total'] + metrics['smooth_l1_loss_{}'.format(k)]


            metrics['orthogonality_loss_{}'.format(k)] = orth_loss * args.lambda_ortho
            metrics['orthogonality_loss_total'] = metrics['orthogonality_loss_total']  + metrics['orthogonality_loss_{}'.format(k)]

            metrics['quantization_loss_{}'.format(k)] = quantization_loss * args.lambda_quant
            metrics['quantization_loss_total'] = metrics['quantization_loss_total'] + metrics['quantization_loss_{}'.format(k)]

    # ipdb.set_trace()

    metrics['total_loss'] = metrics['total_loss'] + metrics['smooth_l1_loss_total']/num_s + \
                            metrics['orthogonality_loss_total']/num_s + \
                            metrics['quantization_loss_total']/num_s

    return metrics, s_hist


def log_vals(logger, val_dict, step, tag='train'):
    print('Training Step: {} '.format(step), end='')
    for name in val_dict.keys():
        val = val_dict[name].item()
        logger.log_scalar(tag='{}/{}'.format(tag, name), value=val, step=step)
        print('{}: {:3f} '.format(name, val), end='')
    print()


def log_hist(logger, hist_dict, step, tag='train'):
    for name in hist_dict.keys():
        logger.log_histogram(tag='{}/{}'.format(tag, name), values=hist_dict[name], step=step, bins=10)


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

        imp_vectors = sup_net(random_one_hot)
        out = net.forward_check_multi(inputs, imp_vectors, switch_vec)
        out_softmax = F.softmax(out, dim=1)

        metrics, s_hist = estimate_metrics(out_softmax, random_query, binary_target, sup_net, switch_vec)

        optimizer.zero_grad()
        metrics['total_loss'].backward()
        optimizer.step()

        if global_step % args.log_after_steps == 0:
            log_vals(logger, metrics, global_step, 'train')
            log_hist(logger, s_hist, global_step, 'train')

        global_step += 1

    return global_step

def val(epoch, global_step=0):
    print('\nEpoch: %d Testing' % epoch)
    sup_net.eval()

    total_metrics = {}
    s_hist = {}

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        one_hot = torch.zeros((inputs.size(0), 10)).fill_(1).to(device)

        random_query = torch.from_numpy(np.random.randint(0, 10, size=(targets.size(0)))).to(device)
        random_one_hot = torch.zeros(inputs.size(0), 10).type(
            torch.cuda.FloatTensor)

        random_one_hot = random_one_hot.scatter_(dim=1, index=random_query.view(-1, 1), src=one_hot)
        random_one_hot = random_one_hot.to(device)
        binary_target = targets.eq(random_query).type(torch.cuda.LongTensor)

        imp_vectors = sup_net(random_one_hot)
        out = net.forward_check_multi(inputs, imp_vectors, switch_vec)
        out_softmax = F.softmax(out, dim=1)

        metrics, s_hist = estimate_metrics(out_softmax, random_query, binary_target, sup_net, switch_vec)

        for name in metrics.keys():
            if name not in total_metrics:
                total_metrics[name] = metrics[name].detach()
            else:
                total_metrics[name] += metrics[name].detach()

    for name in total_metrics.keys():
        total_metrics[name] /= len(testloader)

    log_vals(logger, total_metrics, global_step, 'val')
    log_hist(logger, s_hist, global_step, 'val')

global_step = 0
val(0, global_step=global_step)
for epoch in range(start_epoch+1, start_epoch+201):
    global_step = train(epoch, global_step=global_step)
    val(epoch, global_step=global_step)
    save_path = os.path.join('./checkpoint',args.exp_name, 'models', 'sup_net_epoch_{}.pth'.format(epoch))
    print('Saving model at {}'.format(save_path))
    torch.save(sup_net.state_dict(), save_path)




