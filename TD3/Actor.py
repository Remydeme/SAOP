



from TD3.Model.Actor import ActorModel
import numpy as np



class Actor:

    def __init__(self, action_shape):

        self.actor = ActorModel(action_shape=action_shape)
        self.target_actor = ActorModel(action_shape=action_shape)

        # transfert actor weights into target actor weights
        self.target_actor.set_weights(self.actor.get_weights())




    def get_action(self, state):
        """ Get action compute the next action """
        action = self.actor(state)
        return action

    def get_target_action(self, state):
        """ Compute target action value """
        target_action = self.target_actor(state)
        return target_action

    def copy_to_target(self, tau=0.005):
        target_actor_weigts = self.target_actor.get_weights()
        actor_weights = self.actor_get_weights()
        index = 0
        for ta_weights, a_weihts in zip(target_actor_weigts, actor_weights):
            ta_weights = ta_weights * ( 1 - tau ) + a_weihts * tau
            target_actor_weigts[index] = ta_weights
            index = index + 1

        self.target_actor.set_weights(target_actor_weigts)