import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):

    hidden_layer = 250

    def __init__(self, state_dim, action_dim):
        super().__init__()



        # Actor
        self.l1 = nn.Linear(in_features=state_dim, out_features=self.hidden_layer)
        self.l2 = nn.Linear(in_features=self.hidden_layer, out_features=action_dim)




    def forward(self, state):
        """ Compute V(S) and Q(S,A) """
        state = torch.from_numpy(state).float()

        x = F.relu(self.l1(state))
        Q = F.softmax(self.l2(x), dim=0)


        return Q



class Critic(nn.Module):

    hidden_layer = 250


    def __init__(self, state_dim):
        super().__init__()
        # Critic
        self.l1 = nn.Linear(in_features=state_dim, out_features=self.hidden_layer)
        self.l2 = nn.Linear(in_features=self.hidden_layer, out_features=1)


    def forward(self, state):
        state = torch.from_numpy(state).float()
        x = F.relu(self.l1(state))
        V = self.l2(x)

        return V