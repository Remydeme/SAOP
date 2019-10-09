


import torch.nn as nn
import torch.nn.functional as F
import torch


class CriticModel(nn.Module):

    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, action_dim, state_dim):

        super(CriticModel, self).__init__()

        self.l1 = nn.Linear(action_dim + state_dim, self.hidden_1)

        self.l2 = nn.Linear(self.hidden_1, self.hidden_2)

        self.l3 = nn.Linear(self.hidden_2, 1)



    def forward(self, state, action):

        input = torch.cat([state, action], 1)

        x = F.relu(self.l1(input))

        x = F.relu(self.l2(x))

        x = self.l3(x)

        return x

