import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from supervisor.sup_class import SupervisorQuery


class SupervisorNetwork(nn.Module):

    def __init__(self, path_dims):
        super().__init__()

        self.supervisor_modules = nn.ModuleList()
        self.path_dims = path_dims

        for i_path, path_dim in enumerate(path_dims):
            # This creates a supervisor network for each layer
            if path_dim == 'M':
                # Just copy previous path dim supervi
                self.supervisor_modules.append(None)
            else:
                self.supervisor_modules.append(SupervisorQuery(path_dim))

    def forward(self, class_vec):
        '''
        :params classvec: is the class input vector for the supervisor network
        '''
        # Todo: Here, we calculate importance vectors even for switch 0 layers

        paths = []
        for sup_module in self.supervisor_modules:
            if sup_module == None:
                paths.append(None)
            else:
                paths.append(sup_module(class_vec))

        return paths