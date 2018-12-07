from base_agent import BaseAgent
import numpy as np
import random

class ZeroI(BaseAgent):
    def __init__(self, id, budget, alpha=1, beta=1):
        self.id = id
        self.budget = budget
        self.belief = random.betavariate(alpha, beta)

    def __repr__(self):
        return 'ZeroI id {} belief {} budget {}'.format(self.id, self.belief, self.budget)

    def update_prior(self, signal):
        pass

    def cur_belief(self):
        return self.belief

    def calc_quantity(self, market):
        buying = random.choice([True, False])
        if buying and market.pos_price() < self.cur_belief():
            return np.array([1.0, 0.0])
        elif not buying and market.neg_price() < (1-self.cur_belief()):
            return np.array([0.0, 1.0])
        else:
            return np.array([0.0, 0.])
