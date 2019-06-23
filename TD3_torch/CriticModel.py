

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np






class CriticModel(nn.Module):
    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, state_dim, action_dim):

        super(CriticModel, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_1)
        self.l2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.l3 = nn.Linear(self.hidden_2, 1)


        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_1)
        self.l5 = nn.Linear(self.hidden_1, self.hidden_2)
        self.l6 = nn.Linear(self.hidden_2, 1)



    def forward(self, state, action):

        inputs = torch.cat([state, action], 1)

        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        Q1 = self.l3(x)

        x = F.relu(self.l4(inputs))
        x = F.relu(self.l5(x))
        Q2 = self.l6(x)

        return Q1, Q2


    def Q1(self, state, action):

        inputs = torch.cat([state, action], 1)

        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        Q1 = self.l3(x)
        return Q1