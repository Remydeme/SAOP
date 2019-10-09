import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from ActorCritic.Model import Actor, Critic
from ActorCritic.batch import Batch
import atari_py
import roboschool
from datetime import datetime
import gym




class Agent:

    def __init__(self, env, game, critic_lr=1e-3, actor_lr=1e-3, buffer_size=1000000):

        self.actor = Actor(state_dim=env.observation_space.shape[0], action_dim=4)
        self.critic = Critic(state_dim=env.observation_space.shape[0])

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.actor.parameters(), lr=critic_lr)

        self.loss = torch.nn.CrossEntropyLoss

        self.game = game

        current_time = datetime.now().strftime("%Y%m%d%H")

        name = game + current_time

        self.summary = SummaryWriter(logdir=name)

        self.batch = Batch()






    def take_action(self, state, exploration=False):
        """
        Compute the V(S) and the newt action base on the current state
        :param state:
        :param exploration:
        :return: action, value
        """
        print(state)

        action = self.actor(state)

        value = self.critic(state)

        # return action
        return action, value


    def train(self, gamma=0.99):

        # Training mode  not useful here because we don't use layer that need this type of activation
        #self.actor.train()
        #self.critic.train()

        logs, values, rewards = self.batch.get()

        # Compute all the Qvalues
        q_vals = np.zeros_like(values)

        for t in range(len(reversed(values))):

            # optimal bellman equation
            QVal = rewards[t] + gamma * QVal
            q_vals[t] = QVal

        # We must convert all the numpy array to tensor
        values = torch.from_numpy(values).float()
        logs = torch.from_numpy(logs).float()
        q_vals = torch.from_numpy(q_vals).float()

        # We want to compute the advantage the advantage A = Value - Q(S, A)
        # Because we only have The value V(s) we use the optimal bellman equation to compute Q(S,A)
        # Q(S,A) = R(S,A) + GAMMA * Better(V(S',A'))
        # QVals contains all the Qvalue from the beginning

        advantage = values - q_vals

        # No we know that to update our critic we must compute the gradient of
        # log(Pi(S)) * TD => log(Pi) * R(S, A) + GAMMA * V(S', A') - V(S)

        actor_loss = torch.mean(-logs * advantage)

        # to update the critic we must compute the MSE between the target and the Value
        # TD => sqrt(mean(R(S, A) + GAMMA * V(S', A') - V(S))^2)
        critic_loss = 0.5 * advantage.pow(2).mean()

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.critic_optimizer.step()
        self.actor_optimizer.step()


    def save(self, directory='model', filename='game'):
        torch.save(self.actor.state_dict(), './{}/{}_actor.pth'.format(directory, self.game))
        torch.save(self.critic.state_dict(), './{}/{}_critic.pth'.format(directory, self.game))





def play(env, agent, game, episodes=1000, max_steps=1000, eval_episode_frequency=10):
    """
    Gaming loop
    :return:
    """
    for episode in episodes:

        agent.batch.clean()
        state = env.reset()
        for steps in max_steps:

            logits, value = agent.take_action(state=state)

            value = value.detach().numpy()[0,0]
            logits = action.detach().numpy()

            # here 4 is the number of possible actions
            # sample a random action base on the probability in the array logits
            # It Will return a number between [0, 3] base on the logits probability
            action = np.random.choice(4, p=np.squeeze(logits))

            # compute the entropy here normaly
            next_state, reward, done, _ = env.step(action)

            # store in the batch for future training
            agent.batch.append(logits, value, reward)

            state = next_state

            if done or (steps == max_steps):

                agent.batch.compute_stats_and_display(summary=agent.summary)

                agent.batch.clean_stats_buffer()

                agent.save()

        agent.train()








if __name__ == "__main__":
    env = gym.make('Breakout-ram-v0')
    # reset the env
    a = env.reset()
    s, r, d, _ = env.step([1])

    print(env.observation_space.shape)
    print(s)
    agent = Agent(env=env, game='pong', critic_lr=1e-3, actor_lr=1e-3)

    action = agent.take_action(state=s)

    print(action)