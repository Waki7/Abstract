import torch
import torch.nn as nn
import torch.nn.functional as F

import settings
from networks.base_networks import BaseNetwork
from networks.factory import register_network


@register_network
class ConceptNetwork(BaseNetwork):
    def __init__(self, n_features, out_shape, cfg, **kwargs):
        super(ConceptNetwork, self).__init__(cfg)
        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.model_size = cfg.get('model_size', 32)
        self.bias = cfg.get('bias', True)
        self.n_concepts = cfg.get('n_concepts', 20)
        self.concept_embedding_size = self.n_concepts + out_shape
        self.n_actions = out_shape
        reward_size = 1

        ##########################################################################################
        # create layers
        ##########################################################################################
        self.fc1 = nn.Linear(in_features=self.concept_embedding_size + n_features, out_features=self.model_size)

        self.fc2 = nn.Linear(in_features=self.model_size, out_features=self.model_size, bias=self.bias)

        self.fc3 = nn.Linear(in_features=self.model_size, out_features=self.concept_embedding_size)

        # self.fc4_actor = nn.Linear(in_features=self.concept_embedding_size, out_features=self.concept_embedding_size)
        self.fc4_critic = nn.Linear(in_features=self.concept_embedding_size, out_features=reward_size)
        self.fc4_encoder = nn.Linear(in_features=self.concept_embedding_size,
                                     out_features=n_features)

        self.concept_state = torch.zeros((1, self.concept_embedding_size)).to(settings.DEVICE)  # 1 for batch size

        self.empty_concept = torch.zeros((1, self.concept_embedding_size)).to(settings.DEVICE)
        self.empty_input = torch.zeros((1, n_features)).to(settings.DEVICE)

        self.create_optimizer()

    def forward(self, input=None, concept_state=None
                # ,action
                ):
        if input is None:
            input = self.empty_input.detach().clone()
        if concept_state is None:
            concept_state = self.concept_state
        # if action is not None:
        #     action = action - self.n_actions  # offset by the number of env actions
        #     hidden_mask = self.empty_hidden.detach().clone()
        #     hidden_mask[:, action] = 1.0
        #     self.hidden_state = self.hidden_state * hidden_mask

        x = torch.cat([input, concept_state], dim=-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        pre_concept = self.fc3(x)
        concept_probs = F.softmax(pre_concept, dim=-1)

        # action_probs = concept_probs
        action_probs = F.softmax(pre_concept[:, :self.n_actions], dim=-1)

        encoder_estimate = self.fc4_encoder(concept_probs)

        reward = self.fc4_critic(concept_probs)
        self.concept_state = concept_probs

        return action_probs, reward, concept_probs, encoder_estimate,  # one_step estimate, current concept state has not bounds to a distribution, like it doesn't have ot look like a distirbution

    def prune(self):
        self.concept_state = self.concept_state.detach()
        # pass

    def reset_state(self):
        self.concept_state = torch.zeros((1, self.concept_embedding_size)).to(settings.DEVICE)
    # def update_paramters(self):
    #     pass


@register_network
class GatedConceptNetwork(BaseNetwork):
    def __init__(self, n_features, out_shape, cfg, concept_embedding_size, **kwargs):
        super(GatedConceptNetwork, self).__init__(cfg)
        ##########################################################################################
        # set cfg parameters
        ##########################################################################################
        self.model_size = cfg.get('model_size', 32)
        self.bias = cfg.get('bias', True)
        self.concept_embedding_size = concept_embedding_size
        reward_size = 1

        ##########################################################################################
        # create layers
        ##########################################################################################
        self.fc1 = nn.Linear(in_features=self.concept_embedding_size + n_features, out_features=self.model_size)

        self.fc2 = nn.Linear(in_features=self.model_size, out_features=self.model_size, bias=self.bias)

        self.fc3_actor = nn.Linear(in_features=self.concept_embedding_size, out_features=1)
        self.fc4_critic = nn.Linear(in_features=self.concept_embedding_size, out_features=reward_size)
        self.out_fcs = nn.ModuleList()
        for channel_idx in range(0, self.n_out_channels):
            new_layer = nn.Linear(
                in_features=self.cmbn_fc2.out_features, out_features=self.out_shapes[channel_idx], bias=self.bias)
            self.out_fcs.append(new_layer)

        self.hidden_state = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)  # 1 for batch size

        self.empty_hidden = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)

        self.create_optimizer()

    def forward(self, env_input, reward=None, hidden_input=None, action=None):
        if hidden_input is None:
            hidden_input = self.hidden_state
        if action is not None:
            action = action - self.n_actions  # offset by the number of env actions
            hidden_mask = self.empty_hidden.detach().clone()
            hidden_mask[:, action] = 1.0
            self.hidden_state = self.hidden_state * hidden_mask
        channel_inputs, layer1 = [], []

        input = torch.cat((env_input, hidden_input, reward), dim=-1)
        for channel_idx in range(0, len(self.in_channels)):
            fc1 = self.fc1s[channel_idx]
            channel_input = input[:, self.in_vector_idx[channel_idx]: self.in_vector_idx[channel_idx + 1]]
            layer1.append(fc1(channel_input))

        l1_cmbn = F.leaky_relu(torch.cat(layer1, dim=-1))
        l2 = F.leaky_relu(self.cmbn_fc2(l1_cmbn))
        ly = []

        for channel_idx in range(0, len(self.out_channels)):
            neuron = self.out_fcs[channel_idx]
            outNeuron = neuron(l2)
            ly.append(outNeuron)
            # ly[outputChannel] = F.softmax(outNeuron, dim=1)

        output = torch.cat(ly, dim=1)

        prob_output = F.softmax(output,
                                dim=-1)  # todo try to experiment with having all of the output part of hidden state.

        action = prob_output[:, :self.out_vector_idx[1]]
        hidden_concepts = prob_output[:, self.out_vector_idx[1]:self.out_vector_idx[2]]

        self.hidden_state = hidden_concepts

        return action, hidden_concepts

    def prune(self):
        self.hidden_state = self.hidden_state.detach()
        # pass

    def reset_state(self):
        self.hidden_state = torch.zeros((1, self.hidden_in_shape)).to(settings.DEVICE)
    # def update_paramters(self):
    #     pass
