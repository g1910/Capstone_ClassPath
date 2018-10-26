'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import ipdb


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ClassPriorModule(nn.Module):

    def __init__(self, num_channels, reduction=16):
        super(ClassPriorModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(2*num_channels, 2*num_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(2*num_channels // reduction, num_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels),
            nn.Tanh()
        )

    def forward(self, act, class_prior):

        b, c, _, _ = act.size()
        act_avg = self.avg_pool(act).view(b, c)

        input_x = torch.cat((act_avg, class_prior), dim=1) # b x 2c
        attention_vector = self.attention(input_x) # b x c

        imp_vector = F.relu(torch.mul(attention_vector, class_prior), inplace=True)
        imp_vector = imp_vector.view(b, c, 1, 1)

        return imp_vector * act, torch.sum(torch.abs(attention_vector), dim=1).mean()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.class_prior_modules = nn.ModuleList()

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.class_prior_modules.append(ClassPriorModule(planes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x):
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return out

    def forward_check(self, x, imp_vector):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        count = 1
        for i, layer in enumerate(self.layer2):
            out = layer(out)
            if i == count:
                out = out * imp_vector.unsqueeze(2).unsqueeze(3)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def forward(self, x, class_priors, switch_vector, reg=False):
        out = F.relu(self.bn1(self.conv1(x)))
        l1_reg = 0.
        layer_num = 0
        layer_count = [1,2,3,4]
        for lc in layer_count:
            layer_group = getattr(self, 'layer{}'.format(lc))
            for layer in layer_group:
                out = layer(out)
                if switch_vector[layer_num]:
                    out, l1_act = self.class_prior_modules[layer_num](out, class_priors[layer_num])
                    if reg:
                        l1_reg += l1_act
                layer_num += 1

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, l1_reg

    def forward_with_paths(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        true_correct_paths = []
        true_incorrect_paths = []
        layer_count = [1, 2, 3, 4]
        for lc in layer_count:
            layer_group = getattr(self, 'layer{}'.format(lc))
            for layer in layer_group:
                out = layer(out)
                out_path = torch.mean(torch.mean(out, dim=3), dim=2)
                flipped_path = self.get_flipped_path(out_path)

                true_correct_paths.append(out_path)
                true_incorrect_paths.append(flipped_path)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, true_correct_paths, true_incorrect_paths

    def get_flipped_path(self, input_path):
        mean_path = torch.mean(input_path, dim=1, keepdim=True)
        flipped_path = input_path + 2*(mean_path - input_path)
        return flipped_path

    def pruned_forward_pass_classpath(self, x, class_path, test_vector, compression, desc=False, random=False, reverse=False):
        out = F.relu(self.bn1(self.conv1(x)))
        paths = []
        block_num = 0
        for layer in self.layer1:
            out = layer(out)
            if test_vector[block_num]:
                out = self.prune_activation_channels(out, compression, pred_path=class_path[block_num], desc=desc, random=random, reverse=reverse)
            pred_path = torch.mean(torch.mean(out, dim=3), dim=2)
            paths.append(pred_path)
            block_num += 1

        for layer in self.layer2:
            out = layer(out)
            if test_vector[block_num]:
                # out = out + 2 * (torch.mean(out, dim=1, keepdim=True) - out)
                out = self.prune_activation_channels(out, compression, pred_path=class_path[block_num], desc=desc, random=random, reverse=reverse)
            pred_path = torch.mean(torch.mean(out, dim=3), dim=2)
            paths.append(pred_path)
            block_num += 1

        for layer in self.layer3:
            out = layer(out)
            if test_vector[block_num]:
                # out = out + 2 * (torch.mean(out, dim=1, keepdim=True) - out)
                out = self.prune_activation_channels(out, compression, pred_path=class_path[block_num], desc=desc, random=random, reverse=reverse)
            pred_path = torch.mean(torch.mean(out, dim=3), dim=2)
            paths.append(pred_path)
            block_num += 1

        for layer in self.layer4:
            out = layer(out)
            if test_vector[block_num]:
                # out = out + 2 * (torch.mean(out, dim=1, keepdim=True) - out)
                out = self.prune_activation_channels(out, compression, pred_path=class_path[block_num], desc=desc, random=random, reverse=reverse)
            pred_path = torch.mean(torch.mean(out, dim=3), dim=2)
            paths.append(pred_path)
            block_num += 1

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, paths

    def prune_activation_channels(self, act, compression, pred_path=None, desc=False, random=False, reverse=False):
        """
        Zeros out channels based after sorting tp weights
        """
        # ipdb.set_trace()
        if pred_path is None:
            pred_path = torch.mean(torch.mean(act, dim=3), dim=2)
        else:
            pred_path = Variable(pred_path.type(torch.cuda.FloatTensor)).cuda()

        if reverse:
            pred_path = pred_path + 2 * (torch.mean(pred_path, dim=1, keepdim=True) - pred_path) # flip about mean

        _, sort_idx_pred = torch.sort(pred_path, dim=1, descending=desc)
        num_dropped = int((pred_path.shape[1] * compression) // 4)  # check what len returns: error
        if num_dropped > 0:
            if random:
                sort_idx_pred = sort_idx_pred[:, torch.randperm(pred_path.shape[1])[:num_dropped]] # random
            else:
                sort_idx_pred = sort_idx_pred[:, :num_dropped]
            sort_idx_pred = torch.unsqueeze(sort_idx_pred, 2)
            sort_idx_pred = torch.unsqueeze(sort_idx_pred, 3)

            sort_idx_pred = sort_idx_pred.repeat(1, 1, act.shape[2], act.shape[3])
            zero_map = torch.autograd.Variable(
                torch.zeros((act.shape[0], sort_idx_pred.shape[1], act.shape[2], act.shape[3]))).cuda()
            act = act.scatter_(dim=1, index=sort_idx_pred, src=zero_map)
        return act



def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
