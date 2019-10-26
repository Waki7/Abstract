import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.NetworkTypes import *
import gym_life.envs.life_channels as ch
import numpy as np
import settings

class ChannelNetwork(nn.Module):
    def __init__(self):
        super(ChannelNetwork, self).__init__()
