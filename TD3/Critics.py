





from TD3.Model.Critics import CriticsModel
import tensorflow as tf





class Critics:


    def __init__(self, lr=1e-3):

        self.critic = CriticsModel()

        self.target_critic = CriticsModel()

        # optimizer

        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

        # copy crititc weights to target critics
        self.target_critic.set_weights(self.critic.get_weights())



# j'ai change l'axe et aussi les label c'est tout
    def get_value(self, state, action):
        inputs = tf.concat([state, action], axis=1)
        value, value_bis = self.critic(inputs)
        return value, value_bis


    def get_target_value(self, state, action):
        inputs = tf.concat([state, action], axis=1)
        value, value_bis = self.target_critic(inputs)
        return value, value_bis

    def get_q1(self, state, action):
        inputs = tf.concat([state, action], axis=1)
        value = self.critic.Q1(inputs)
        return value


    def train(self, state, action, y):
        inputs = tf.concat([state, action], axis=1)
        error = self.critic.train_on_batch(inputs, [y, y])
        return error


    def copy_to_target(self, tau=0.005):
        target_critic_weights = self.target_critic.get_weights()
        critic_weights = self.critic.get_weights()
        index = 0
        for tc_weights, c_weights in zip(target_critic_weights, critic_weights):
            tc_weights = tc_weights * (1 - tau) + c_weights * tau
            target_critic_weights[index] = tc_weights
            index = index + 1

        self.target_critic.set_weights(target_critic_weights)

