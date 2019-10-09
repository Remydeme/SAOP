
import torch
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

import roboschool
from DDPG.Models.ActorModel import ActorModel
from DDPG.Models.CriticModel import CriticModel
from TD3_torch.ReplayBuffer import ReplayBuffer

from datetime import datetime

import gym

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, env, game, critic_lr=1e-3, actor_lr=1e-3, buffer_size=1000000):

        self.env = env
        self.game = game
        self.actor = ActorModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], action_max=env.action_space.high[0]).to(device)
        self.actor_target = ActorModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], action_max=env.action_space.high[0]).to(device)

        self.critic = CriticModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(device)
        self.critic_target = CriticModel(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.replay_buffer = ReplayBuffer(max_size=buffer_size)


        self.critic_loss_array = []

        self.q_score_array = []

        current_time = datetime.now().strftime("%Y%m%d%H")

        name = game + current_time

        self.summary = SummaryWriter(name)


    def take_action(self, state, noise=0.1, noisy_action=False):
        """ Function that return an action  base on the current state

            Args:

              state: current state
              noise: the value of the stddev of the gaussian noise that is added to the action
              noisy_action: Boolean that indicate if we add noise to the action

           Returns:
              The optimal action that permit to get the maximum reward
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if noisy_action:

            noise = np.random.normal(loc=0, scale=noise, size=self.env.action_space.shape[0])
            action = action + noise
            action = action.clip(self.env.action_space.low, self.env.action_space.high)
            return action

        else:

            return action

    def updateNetworks(self, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2):
        """

        :param iterations: Number of steps made by the agent.
        :param batch_size: Size of the sample use to train the agent
        :param discount: discount parameter to specify the importance of future action
        :param tau: Polyak constant use to compute the polyak average or soft copy
        :param policy_noise: Noise add to actor action for to explore the environment
        :return:
        """
        for it in range(iterations):

            x, y, a, r, d = self.replay_buffer.sample(batch_size=batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)


            next_action = self.actor_target(next_state)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)



            # Optimize the critic
            """
                    Since the backward() function accumulates gradients, and you don’t want 
                    to mix up gradients between minibatches, you have to zero them out at the start of a new minibatch.
            """
            self.critic_optimizer.zero_grad() # zero the gradient
            critic_loss.backward()
            self.critic_optimizer.step()

            # store the error
            self.critic_loss_array.append(critic_loss.detach().numpy())


            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()



            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # store the critic score
            self.q_score_array.append(actor_loss.detach().numpy())

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def error_and_score_log(self, episode):
        """ Write the error to tensorboard """
        mean_critic_error = np.mean(self.critic_loss_array)
        mean_critic_score = np.mean(self.q_score_array)

        self.summary.add_scalar("critic score", mean_critic_score, episode)
        self.summary.add_scalar("critic error", mean_critic_error, episode)

        # clean  our array
        self.critic_loss_array.clear()
        self.q_score_array.clear()

    def save(self, filename, directory):
        #torch.save(self.actor.state_dict(), '%s_actor.pth' % (filename))
        #torch.save(self.critic.state_dict(), '%s_critic.pth' % (filename))
        torch.save(self.actor.state_dict(), './cheetah_actor.pth')
        torch.save(self.critic.state_dict(), './cheetah_critic.pth')



def evaluateModel(env, agent, episode):
    for e in range(100):
        state = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            a = agent.take_action(state=state, noisy_action=False)
            state, r, done, _ = env.step(a)
            ep_score += r
        ep_memory.append(ep_score)
    evaluation_score = np.mean(ep_memory)
    agent.summary.add_scalar('evaluation score', evaluation_score, episode)
    return evaluation_score

def play(agent, env, game):
    scores = []
    training_delay = 1000
    step = 0
    episode = 0
    done = False
    state = env.reset()
    episode_iterations = 0
    episode_score = 0
    best_evaluation_score = 0
    evaluation_delay = 5000   # number of episode between each evaluation
    while step < 3000000:
        if done:
            agent.updateNetworks(iterations=episode_iterations)
            agent.summary.add_scalar("episode score", episode_score, step)
            agent.summary.add_scalar("step by episode", episode_iterations, episode)
            scores.append(episode_score)
            agent.error_and_score_log(episode=episode)
            print("Episode : {} score {} ".format(episode, episode_score))
            episode_score = 0
            state = env.reset()
            episode += 1
            episode_iterations = 0
        # trick use by OPenAI we train the policy on random gaussian action
        if training_delay < step:
            action = agent.take_action(state=state, noisy_action=True)
        else:
            action = env.action_space.sample()

        next_state, reward, done, _ = env.step(action)

        episode_score += reward

        agent.replay_buffer.add((state, next_state, action, reward,  done))

        state = next_state

        episode_iterations += 1

        step += 1

        if step % evaluation_delay == 0 and episode != 0:
            eval_score = evaluateModel(env=env, agent=agent, episode=episode)
            print("epsiode {} average score {} ".format(episode, eval_score))
            if eval_score > best_evaluation_score:
                best_evaluation_score = eval_score
                directory = "./model_saved/" + game
                agent.save(filename=game, directory=directory)


    return agent



if __name__ == "__main__":
    game = "RoboschoolHalfCheetah-v1"
    env = gym.make(game)
    env.seed(seed=0)
    agent = Agent(env=env, game=game)
    agent = play(agent=agent,env=env, game=game)
    env.reset()
    for e in range(100):
        # reset the enviroment
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            a = agent.take_action(state=s, noisy_action=False)
            s, r, done, _ = env.step(a)
            env.render()