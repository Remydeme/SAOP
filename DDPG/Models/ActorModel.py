

import torch
import torch.nn as nn
import torch.nn.functional as F




class ActorModel(nn.Module):

    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, state_dim, action_dim, action_max):

        super(ActorModel, self).__init__()

        self.action_max = action_max

        self.l1 = nn.Linear(state_dim, self.hidden_1)

        self.l2 = nn.Linear(self.hidden_1, self.hidden_2)

        self.l3 = nn.Linear(self.hidden_2, action_dim)


    def forward(self, input):

        x = F.relu(self.l1(input))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.action_max
        return x
