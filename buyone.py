from base_agent import BaseAgent
import logging
import numpy as np

class BuyOne(BaseAgent):
    # def __repr__(self):
    #     return 'BuyOne id {} budget {} belief {}'.format(self.id, self.budget, self.cur_belief())
    def name(self):
        return 'BuyOne'

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
