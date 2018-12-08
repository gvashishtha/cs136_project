import random

class BaseAgent(object):
    # BaseAgent - other agent classes should subclass this agent
    def __init__(self,
        id, budget, true_alpha=1, true_beta=1, noise=0., alpha=1, beta=1):
        self.id = id
        self.budget = budget
        self.alpha = alpha # ignore true alpha and beta values
        self.beta = beta
        self.noise = noise

    def __repr__(self):
        # Return a string representing this agent
        return 'Agent {} id {}'.format(self.name(), self.id)

    def update_prior(self, signal):
        # Given a 0, 1 signal from the underlying Beta distribution,
        # update the agent's prior beliefs about the likelihood of the event
        if signal == 1:
            if random.random() < self.noise: # sometimes, do the wrong update
                self.beta += 1
            else:
                self.alpha += 1
        elif signal == 0:
            if random.random() < self.noise: # sometimes, do the wrong update
                self.alpha += 1
            else:
                self.beta += 1

    def cur_belief(self):
        # Should return a float in [0, 1] that represents the belief of
        # the agent about the likelihood of the event's ocurring
        return float(self.alpha)/(float(self.alpha)+float(self.beta))

    def calc_quantity(self, market):
        # Should return a numpy array [x, y] where x represents the quantity
        # of q0 we wish to buy and y represents the quantity of q1 we wish
        # to buy
        raise NotImplementedError
