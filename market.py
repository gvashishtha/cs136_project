import copy
import logging
import math
import numpy as np

class LMSRMarket(object):
    def __init__(self, state=None, alpha=1.0, beta=1.0):
        if state is None:
            # state is a numpy array
            self.state = np.array([0.,0.])
        else:
            self.state = copy.deepcopy(state)
        self.beta = beta
        self.revenue = 0.0

    def __repr__(self):
        return 'LMSR market with state {} revenue {}'.format(self.state, self.revenue)

    def get_price(self, trade):
        # Tells us how much to charge the agent for a given trade
        try:
            # print("Self.get_cost(state) is: ", self.get_cost(self.state))
            return self.get_cost(self.state + trade) - self.get_cost(self.state)
        except ValueError:
            print 'ValueError trade is {} state is{}'.format(trade, self.state)
            raise ValueError

    def trade(self, trade):
        # updates the market state based on a given trade
        self.revenue += self.get_price(trade)
        self.state += trade

    def get_cost(self, state):
        # Uses LMSR scoring to calculate cost under a given state (eqn 18.5)
        cost = self.beta * math.log(sum(map(lambda x: math.exp(x/self.beta), state)))
        logging.debug('cost is {} for state {}'.format(cost, state))
        return cost

    def instant_price(self, index):
        # Use formula 18.6 in textbook
        # Calculates the instantaneous price for contract at index
        powers = map(lambda x: math.exp(x/self.beta), self.state)
        return math.exp(self.state[index]/self.beta)/sum(powers)

    def pos_price(self):
        #return self.get_price([1,0])
        return self.instant_price(0)

    def neg_price(self):
        return self.instant_price(1)
        #return self.get_price([0,1])

    def calc_quantity(self, true_belief):
        # given an agent's true belief, calculate a vector of trades
        # that would make the market belief match their true belief
        if self.pos_price() < true_belief:
            index = 0
        elif self.neg_price() < (1.-true_belief):
            index = 1
            true_belief = 1. - true_belief # we are trying to reset the negative belief
        else:
            return np.array([0.0, 0.0])

        constant = math.exp(self.state[1-index]/self.beta)
        try:
            quant = self.beta*math.log(true_belief * constant/(1.-true_belief))-float(self.state[index])
        except ZeroDivisionError:
            quant = 0.

        out = np.array([0.0, 0.0])
        out[index] = quant
        return out


class LMSRProfitMarket(LMSRMarket):
    """
    Modified LMSR to turn profit by relaxing normalization
    """
    def __init__(self, state=np.array([0.,0.]), beta=1.0, alpha=1.0):
        LMSRMarket.__init__(self, state, beta)
        if np.array_equal(self.state,np.array([0.,0.])):
            self.state += np.array([0.1, 0.1])
        elif sum(self.state) <= 0:
            logging.error('state set incorrectly')
            raise ValueError

        self.alpha = alpha
        if self.alpha <= 0:
            logging.error('alpha must be greater than 0')
            raise ValueError


    def update_beta(self):
        """
        Given quantity vector (market state) q, we have:
            beta(q) = alpha*sum_i(q_i)
        """
        self.beta = self.alpha*sum(self.state)
        if self.beta == 0:
            logging.error('LMSRProfit has beta 0')
            raise ValueError

    def instant_price(self, index):
        """
        Use formula for price at state i from 4.1 in Othman paper
        Calculates the instantaneous price for contract at index
        """
        # update beta
        self.update_beta()
        powers = map(lambda x: math.exp(x/self.beta), self.state)

        # price at index i = A + ((B - C)/D)
        A = self.alpha*math.log(sum(powers))
        B = sum(self.state*math.exp(self.state[index]/self.beta))
        C = sum(map(lambda x: x*math.exp(x/self.beta), self.state))
        D = sum(self.state)*sum(powers)

        # if we are dividing by zero, no contracts bought
        # so we use original LMSR
        if D == 0:
            return LMSRMarket().instant_price(index)

        return (A + (B - C)/D)


    def get_cost(self, state):
        # update beta
        self.update_beta()
        # Uses LMSR to calculate cost under a given state (eqn 18.5)
        return LMSRMarket(self).get_cost(state)
