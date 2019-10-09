"""

@author: Remy DEME
@copyright : All rights reserved

^^

"""




import tensorflow as tf

from TD3.Actor import Actor

from TD3.Critics import Critics

from TD3.replayBuffer import ReplayBuffer

import numpy as np

import roboschool

import gym

from datetime import datetime

class TD3Agent():



    def __init__(self, env, game, critic_lr=1e-3, actor_lr=1e-3, target_action_noise=0.2, action_noise=0.1,  buffer_size=1000000):

        self.env = env
        self.game = game
        self.actor_noise = action_noise
        self.target_action_noise = target_action_noise
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.actor = Actor(action_shape=env.action_space.shape[0])
        self.critic = Critics(lr=critic_lr)
        self.replayBuffer = ReplayBuffer(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0], size=buffer_size)

        # array to store the error

        self.critic_score = []
        self.critic_error = []

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)


    def updateNetworks(self,steps, batch_size=100, gamma=0.99, noise_clip=0.5):

        for step in range(steps):

            state, next_state, action, reward, done = self.replayBuffer.sample_batch(batch_size=batch_size)

            state = tf.convert_to_tensor(state, dtype=tf.float64)
            next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
            action = tf.convert_to_tensor(action, dtype=tf.float64)
            reward = tf.convert_to_tensor(reward, dtype=tf.float64)
            done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)

            # compute future action
            next_action = self.actor.get_target_action(state=state)

            # add noise to the action
            epsilon = tf.random.normal(shape=tf.shape(next_action), stddev=self.target_action_noise, dtype=tf.float64)
            target_action_noise = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
            next_action = next_action + target_action_noise
            next_action = tf.clip_by_value(next_action, self.env.action_space.low[0], self.env.action_space.high[0])

            # compute the target value

            t_value, t_value_bis = self.critic.get_target_value(next_state, next_action)

            # take the min value between the two to compute the target value. It permit to reduce the variance.
            # We have a lower variance for the critic value so we have a better critic value and our actor is by the way
            # better

            min_value = tf.minimum(t_value, t_value_bis)

            target_value = reward + (1 - done) * gamma * min_value

            # train critics

            critic_error = self.critic.train(state, action, target_value)

            self.critic_error.append(critic_error)


            # we update the policy and the weights of the target lass than the critics

            if step % 2 == 0:

                with tf.GradientTape() as tape:
                    actor_action = self.actor.get_action(state=state)
                    # compute the score gave by the critic fopr the action => R(S, A)
                    # Q1(S, A) is used but we can use Q2(S,A) as critic score
                    c_value = self.critic.get_q1(state, actor_action)
                    c_value = - tf.reduce_mean(c_value)
                gradient = tape.gradient(c_value, self.actor.actor.trainable_variables)

                # apply the gradient
                self.optimizer.apply_gradients(zip(gradient, self.actor.actor.trainable_variables))

                # store the score
                self.critic_score.append(c_value)

                # update the target networks
                self.actor.copy_to_target()
                self.critic.copy_to_target()


    def get_action(self, state):
        """ Return the best action for the given state and add noise to the action """

        action = self.actor.get_action(state=state)

        action = action[0] * self.env.action_space.high[0]
        # add noise
        action = action + np.random.normal(0, self.actor_noise, size=self.env.action_space.shape[0])

        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action


    def take_action(self, state):
        """ Return the best action for the given state"""
        action = self.actor.get_action(state=state)

        return action[0] * self.env.action_space.high[0]



    def error_and_score_log(self, step):
        """ Write the error to tensorboard """
        mean_critic_score = np.mean(self.critic_score)
        mean_critic_error = np.mean(self.critic_error)

        tf.summary.scalar("critic score", mean_critic_score, step=step)
        tf.summary.scalar("critic error", mean_critic_error, step=step)

        # clean  our array
        self.critic_error.clear()
        self.critic_score.clear()


    def save_model(self, critic_file_name, actor_file_name):
        # add date extension to the file name
        date = datetime.now().strftime("%Y%m%d-%H")

        # model file name
        critic_file_name = "memory/" + self.game + "/" + critic_file_name
        actor_file_name = "memory/" + self.game + "/" + actor_file_name
        critic_file_name = critic_file_name + "_" + self.game  + "_" + date
        actor_file_name = actor_file_name + "_" + self.game + "_" + date

        self.critic.critic.save_weights(critic_file_name)
        self.critic.target_critic.save_weights(critic_file_name + "_target")

        # save the actor model

        self.actor.actor.save_weights(actor_file_name)
        self.actor.target_actor.save_weights(actor_file_name + "_target")







def evaluateModel(env, agent, episode):
    for e in range(100):
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            s = s.reshape([1, env.observation_space.shape[0]])
            a = agent.take_action(state=s)
            s, r, done, _ = env.step(a.numpy())
            ep_score += r
        ep_memory.append(ep_score)
    evaluation_score = np.mean(ep_memory)
    tf.summary.scalar('evaluation score', evaluation_score, step=episode)







def play(agent, env, game):
    current_time = datetime.now().strftime("%Y%m%d%H")
    train_log_dir = 'logs/gradient_tape/' + "Agent" + game + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    scores = []
    batch_size = 100
    training_delay = 1000
    step = 0
    e = 0
    done = False
    s = env.reset()
    episode_iterations = 0
    ep_score = 0
    evaluation_delay = 5000   # number of episode between each evaluation
    while step < 3000000:
        if done:
            agent.updateNetworks(steps=episode_iterations, batch_size=batch_size)
            with train_summary_writer.as_default():
                agent.error_and_score_log(step=step)
                tf.summary.scalar('episode score', ep_score, step=e)
                tf.summary.scalar('step by episode', episode_iterations, step=e)
                scores.append(ep_score)
            print("episode {} | step {} | score {}".format(e, step, ep_score))
            ep_score = 0
            agent.save_model(critic_file_name="critic", actor_file_name="actor")
            s = env.reset()
            e += 1
            episode_iterations = 0
        # make the chosen action
        s = s.reshape([1, env.observation_space.shape[0]])

        # convert to tensor to be able to compute the action with the neural net
        observation = tf.convert_to_tensor(s, dtype=tf.float64)

        # trick use by OPenAI we train the policy on random gaussian action
        if training_delay < step:
            a = agent.get_action(state=observation)
        else:
            a = env.action_space.sample()

        next_state, r, done, _ = env.step(a)

        ep_score += r

        agent.replayBuffer.store(obs=s, act=a, rew=r, next_obs=next_state, done=done)

        s = next_state

        episode_iterations += 1

        step += 1

        with train_summary_writer.as_default():
            if step % evaluation_delay == 0 and e != 0:
                evaluateModel(env=env, agent=agent, episode=e)

    return agent



if __name__ == "__main__":
    game = "RoboschoolHalfCheetah-v1"
    env = gym.make(game)
    env.seed(seed=0)
    agent = TD3Agent(env=env, game=game)
    agent = play(agent=agent,env=env, game=game)
    env.reset()
    for e in range(100):
        # reset the enviroment
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            s = s.reshape([1, env.observation_space.shape[0]])
            a = agent.take_action(state=s)
            s, r, done, _ = env.step(a.numpy())
            env.render()






















