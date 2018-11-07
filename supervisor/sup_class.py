import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


# LeNet Type Supervisor Model
class SupervisorQuery(nn.Module):
    def __init__(self, path_dim):
        super(SupervisorQuery, self).__init__()
        self.class_features = nn.Sequential(
            nn.Linear(100, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(inplace=True),
        )

        self.path_net = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, path_dim),
            nn.Sigmoid()
        )

    def forward(self, class_vec):
        class_feat = self.class_features(class_vec)
        return self.path_net(class_feat)