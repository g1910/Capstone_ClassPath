import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from supervisor.sup_class import SupervisorQuery


class SupervisorNetwork(nn.Module):

    def __init__(self, path_dims, input_dim=100):
        super().__init__()

        self.supervisor_modules = nn.ModuleList()

        for path_dim in path_dims:
            self.supervisor_modules.append(SupervisorQuery(path_dim, input_dim=input_dim))

    def forward(self, class_vec):

        paths = []
        for sup_module in self.supervisor_modules:
            paths.append(sup_module(class_vec))

        return paths