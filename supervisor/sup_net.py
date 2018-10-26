import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from supervisor.sup_module import SupervisorModule


class SupervisorNetwork(nn.Module):

    def __init__(self, path_dims):
        super().__init__()

        self.supervisor_modules = nn.ModuleList()

        for path_dim in path_dims:
            self.supervisor_modules.append(SupervisorModule(path_dim))

    def forward(self, class_vec, image):

        paths = []
        for sup_module in self.supervisor_modules:
            paths.append(sup_module(class_vec, image))

        return paths