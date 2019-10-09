import numpy as np




class Batch:

    def __init__(self):
        """
        A Batch that store logits, reward and value during the game
        """
        self.logs = []
        self.rewards = []
        self.values = []

        self.logs_stats = []
        self.rewards_stats = []
        self.values_stats = []

    def append(self, log, value, reward):
        """
        Add new set of of logit , value and reward in the batch
        :param log:
        :param value:
        :param reward:
        :return:
        """
        self.logs.append(log)
        self.values.append(value)
        self.rewards.append(reward)

        # will be use for stats
        self.logs_stats.append(log)
        self.values_stats.append(value)
        self.rewards_stats.append(reward)


    def get(self):
        """

        :return: logs, valuen, reward
        """
        return self.logs, self.values, self.rewards


    def compute_stats_and_display(self, summary=None, display=True):
        """
        Compute the mean of rewards, values and logits and return them
        :param display:
        :return: mean_logs, mean_reward, mean_value
        """
        mean_logs = np.mean(self.logs_stats)
        mean_reward = np.mean(self.rewards_stats)
        mean_value = np.mean(self.values_stats)

        if display == True:
            print('Logs : {}, Reward : {}, Values {}'.format(mean_logs, mean_reward, mean_value))

        return mean_logs, mean_reward, mean_value

    def clean(self):
        """
        Clear the buffer
        :return:
        """
        if len(self.logs) != 0:
            self.logs.clear()
            self.rewards.clear()
            self.values.clear()

    def clean_stats_buffer(self):
        """
        Clear the buffer
        :return:
        """
        if len(self.logs) != 0:
            self.logs_stats.clear()
            self.rewards_stats.clear()
            self.values_stats.clear()
