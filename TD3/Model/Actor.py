"""

@author: Remy DEME
@copyright : All rights reserved

^^

"""






import tensorflow as tf
from tensorflow import keras



"""
    Notre acteur est contruit avec le framework tensorflow 2.0 
    On utilise tf.keras.model

"""
class ActorModel(tf.keras.Model):
    """ """

    hidden_one_units = 400

    hidden_two_units = 300

    def __init__(self, action_shape):
        super(ActorModel, self).__init__()

        self.hidden_one = keras.layers.Dense(self.hidden_one_units, activation='relu', name='hidden_one')

        self.hidden_two = keras.layers.Dense(self.hidden_two_units, activation='relu', name='hidden_two')

        self.outputs = keras.layers.Dense(action_shape, activation='tanh', name='action_output')




    def call(self, state):

        x = self.hidden_one(state)

        x = self.hidden_two(x)

        action = self.outputs(x)

        return action

