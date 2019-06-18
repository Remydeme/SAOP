"""

@author: Remy DEME
@copyright : All rights reserved

^^

"""




import tensorflow as tf
from tensorflow import keras


class CriticsModel(keras.Model):

    hidden_one_dim = 400
    hidden_two_dim = 300

    def __init__(self):

        super(CriticsModel, self).__init__()


        # first critic neural net

        self.hidden_layer_one = keras.layers.Dense(self.hidden_one_dim, activation='relu', name='hidden_layer_one')

        self.hidden_layer_two = keras.layers.Dense(self.hidden_two_dim, activation='relu', name='hidden_layer_two')

        #initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

        self.outputs_1 = keras.layers.Dense(1, activation='linear', name='outputs')#, kernel_initializer=initializer)



        # Second network neural net

        self.hidden_layer_one_bis = keras.layers.Dense(self.hidden_one_dim, activation='relu', name='hidden_layer_one')

        self.hidden_layer_two_bis = keras.layers.Dense(self.hidden_two_dim, activation='relu', name='hidden_layer_two')

        #initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)

        self.outputs_2 = keras.layers.Dense(1, activation='linear', name='outputs_bis')#, kernel_initializer=initializer)



    def call(self, input):

        x = self.hidden_layer_one(input)

        x = self.hidden_layer_two(x)

        value = self.outputs_1(x)



        x_bis = self.hidden_layer_one_bis(input)

        x_bis = self.hidden_layer_two_bis(x_bis)

        value_bis = self.outputs_2(x_bis)

        return value, value_bis


    def Q1(self, input):

        x = self.hidden_layer_one(input)

        x = self.hidden_layer_two(x)

        value = self.outputs_1(x)

        return value