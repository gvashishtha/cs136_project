import logging
import numpy as np

class BuyOne():
    def __init__(self, id, budget):
        self.id = id
        self.budget = budget
        self.alpha = 1
        self.beta = 1

    def __repr__(self):
        return 'BuyOne agent with id {} budget {} alpha {} beta {}'.format(self.id, self.budget, self.alpha, self.beta)

    def update_prior(self, signal):
        if signal == 1:
            self.alpha += 1
        else:
            self.beta += 1

    def cur_belief(self):
        return float(self.alpha)/(float(self.alpha)+float(self.beta))

    def calc_quantity(self, market):
        # If market probability is less than current belief, try to buy 1 share for the outcome
        # else if market probability is greater, try to buy 1 share against outcome
        pos_price = market.pos_price()
        neg_price = market.neg_price()
        logging.debug('agent {} current belief {} market pos price {} market neg price {}'.format(self.id, self.cur_belief(), pos_price, neg_price))
        if self.cur_belief() > pos_price:
            return np.array([1.0, 0.0])
        elif (1-self.cur_belief()) > neg_price:
            return np.array([0.0, 1.0])
        else:
            return np.array([0.0, 0.0])
