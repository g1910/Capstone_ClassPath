import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


# LeNet Type Supervisor Model
class SupervisorPolicy(nn.Module):
    def __init__(self, path_dim):
        super(SupervisorModule, self).__init__()
        self.class_features = nn.Sequential(
            nn.Linear(10, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
        )

        self.image_features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.path_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, path_dim),
            nn.Sigmoid()
        )

    def forward(self, class_vec, img):
        class_feat = self.class_features(class_vec)
        image_feat = self.image_features(img).view(img.size(0), -1)

        path_feat = torch.cat((class_feat, image_feat), dim=1)
        return self.path_net(path_feat)