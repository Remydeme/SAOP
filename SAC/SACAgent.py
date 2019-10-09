
from SAC.ValueModel import ValueModel
from SAC.CriticModel import CriticModel
from SAC.ActorModel import ActorModel
from TD3_torch.ReplayBuffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:

    def __init__(self, env, game_name, critic_lr=1e-3, actor_lr=1e-3, value_lr=1e-3):

        self.env = env

        self.actor = ActorModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], action_max=env.action_space.high[0])

        self.valueNet = ValueModel(state_dim=env.observation_space.shape[0])

        self.valueNet_target = ValueModel(state_dim=env.observation_space.shape[0])

        self.criticNet = CriticModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])

        self.valueOptimizer = torch.optim.Adam(self.valueNet.parameters(), lr=value_lr)

        self.criticOptimizer = torch.optim.Adam(self.criticNet.parameters(), lr=critic_lr)

        self.actorOptimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.replayBuffer = ReplayBuffer()

        self.valueLossArray = []

        self.actorLossArray = []

        self.criticLossArray = []



    def updateNetworks(self, iterations, batch_size=100, discount=0.99, tau=0.005):

        x, y, r, a, d = self.replayBuffer.sample(batch_size=batch_size)

        state = torch.FloatTensor(x).to(device)
        next_state = torch.FloatTensor(y).to(device)
        reward = torch.FloatTensor(r).to(device)
        action = torch.FloatTensor(a).to(device)
        done = torch.FloatTensor(d).to(device)

        # get the value return by the critic network compute the critics values

        Q1, Q2 = self.criticNet(state, action)

        # we need to compute the target reward + gamma * V

        # first we compute V(S)

        V = self.valueNet(state)

        # compute V(S')

        target_V = self.valueNet_target(next_state)

        target = reward + discount * target_V

        # now we can compute the error for the critic

        criticLossF = F.mse_loss(target, Q1).to(device) + F.mse_loss(target, Q2)

        # compute the action

        action = self.actor(state)

        # compute the gradient and propagate to parameter
        criticLossF.backward()

        # update our weights with the gradient times the learning weights
        self.criticOptimizer.step()








