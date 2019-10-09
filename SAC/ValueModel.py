import torch.nn as nn

import torch.nn.functional as F




class ValueModel(nn.Module):

    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, state_dim):
        """

        :param state_dim: dimension of the observation input
        """
        super(ValueModel, self).__init__()

        self.l1 = nn.Linear(state_dim, self.hidden_1)

        self.l2 = nn.Linear(self.hidden_1, self.hidden_2)

        self.l3 = nn.Linear(self.hidden_2,  1)



    def forward(self, state):
        """
        Compute the value a the state
        :param state: the agent state
        :return: The value of the state S
        """

        x = F.relu(self.l1(state))

        x = F.relu(self.l2(x))

        V = self.l3(x)

        return V

