

import torch.nn as nn

import torch.nn.functional as F

import torch

from torch.distributions import Normal


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ActorModel(nn.Module):

    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, state_dim, action_dim, action_max, init_w=3e-3, std_min=-20, std_max=2):
        """
        :param state_dim: dimension of observation
        :param action_dim: action dimension. The number of joins that we drive
        :param action_max: the maximum value that an action can take
        """
        self.std_min = std_min

        self.std_max = std_max

        self.action_max = action_max

        self.l1 = nn.Linear(state_dim, self.hidden_1)

        self.l2 = nn.Linear(self.hidden_1, self.hidden_2)

        self.mean = nn.Linear(self.hidden_2, action_dim)

        self.mean.weight.data.uniform_(-init_w, init_w)

        self.mean.bias.data.uniform_(-init_w, init_w)

        self.std_log = nn.linear(self.hidden_2, action_dim)

        self.std_log.weight.data.uniform(-init_w, init_w)

        self.std_log.bias.data.uniform(-init_w, init_w)



    def forward(self, state):
        """

        :param state: current state of the agent
        :return: return the next action
        """
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))

        mean = self.mean(x)

        std = self.std_log(x)

        std = torch.clamp(std, self.std_min, self.std_max)

        return mean, std

    def evaluate(self, state, epsilon=1e-6):

        mean, log_std = self.forward(state)

        std = log_std.exp()

        normal = Normal(0, 1)

        # get a random value
        z = normal.sample()

        # reparemetrization tricks
        action = torch.tanh(mean + std * z.to(device))

        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)

        z = normal.sample().to(device)

        action = torch.tanh(mean + std * z)

        action = action.cpu() # put the action on the CPU and return it

        return action[0]




















