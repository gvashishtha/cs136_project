class BaseAgent(object):
    # BaseAgent - other agent classes should subclass this agent
    def __init__(self, id, budget, alpha=1, beta=1):
        self.id = id
        self.budget = budget
        self.alpha = 1 # ignore true alpha and beta values
        self.beta = 1

    def __repr__(self):
        # Return a string representing this agent
        raise NotImplementedError

    def update_prior(self, signal):
        # Given a 0, 1 signal from the underlying Beta distribution,
        # update the agent's prior beliefs about the likelihood of the event
        raise NotImplementedError

    def cur_belief(self):
        # Should return a float in [0, 1] that represents the belief of
        # the agent about the likelihood of the event's ocurring
        raise NotImplementedError

    def calc_quantity(self, market):
        # Should return a numpy array [x, y] where x represents the quantity
        # of q0 we wish to buy and y represents the quantity of q1 we wish
        # to buy
        raise NotImplementedError
