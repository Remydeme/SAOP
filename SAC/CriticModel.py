




import torch.nn as nn

import torch.nn.functional as F

import torch


class CriticModel(nn.Module):

    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: DImension of the observation
        :param action_dim: Dimension of the action played by the agent
        """
        super(CriticModel, self).__init__()


        self.l1 = nn.Linear(state_dim + action_dim, self.hidden_1)

        self.l2 = nn.Linear(self.hidden_1, self.hidden_2)

        self.l3 = nn.Linear(self.hidden_2, 1)


        self.l4 = nn.Linear(state_dim + action_dim, self.hidden_1)

        self.l5 = nn.Linear(self.hidden_1, self.hidden_2)

        self.l6 = nn.Linear(self.hidden_2, 1)




    def forward(self, state, action):
        """
        Compute the critics values of the couple (S, A) and return Q1 and Q2
        the score of two critic networks.
        :param state: State where the agent played the action A
        :param action: Action played by the agent at the State S
        :return: (Q1, Q2) the critics score for the couple (S, A)
        """

        input = torch.cat([state, action], 1)

        x = F.relu(self.l1(input))

        x = F.relu(self.l2(x))

        Q1 = self.l3(x)

        x = F.relu(self.l4(input))

        x = F.relu(self.l5(x))

        Q2 = self.l6(x)


        return Q1, Q2








