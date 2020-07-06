import torch.nn as nn
import torch.nn.functional as F

from networks.basic_fc_networks import BaseFCNetwork
from networks.net_factory import register_network


@register_network
class GraphActor(BaseFCNetwork):
    def __init__(self, n_features, action_nodes, cfg,
                 in_channels=None, in_shapes=None, out_channels=None, out_shapes=None, **kwargs):
        super(GraphActor, self).__init__(cfg)

        self.linear1 = nn.Linear(n_features, self.model_size)
        self.linear2 = nn.Linear(self.model_size, self.model_size)

        modules = nn.ModuleList()

        for action in action_nodes:
            modules.append(nn.Linear(self.model_size, action.space.n))
            action_nodes.pop(0)
            action_nodes.extend(action_nodes.pop(0).children)
        self.linear2 = nn.Linear(self.model_size, self.model_size)
        self.create_optimizer()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear2(x), dim=-1)
        return x
