import math
import numpy as np

class LMSRMarket():
    def __init__(self, state=np.array([0.,0.]), beta=1.0):
        self.state = state
        self.beta = beta
        self.revenue = 0.0

    def get_price(self, trade):
        #new_state = np.array([self.state[0]+trade[0], self.state[1]+trade[1]])
        return self.get_cost(self.state + trade) - self.get_cost(self.state)

    def trade(self, trade):
        self.revenue += self.get_price(trade)
        self.state += trade
        #[self.state[0]+trade[0], self.state[1]+trade[1]]

    def get_cost(self, state):
        powers = map(lambda x: math.exp(x/self.beta), state)
        # for i in range(len(state))]
        return self.beta * math.log(sum(powers))

    # Use formula 18.6 in textbook
    def instant_price(self, index):
        powers = map(lambda x: math.exp(x/self.beta), self.state)
        #powers = [math.exp(self.state[i]/self.beta) for i in range(len(self.state))]
        return math.exp(self.state[index]/self.beta)/sum(powers)

    def pos_price(self):
        #return self.get_price([1,0])
        return self.instant_price(0)

    def neg_price(self):
        return self.instant_price(1)
        #return self.get_price([0,1])

    def calc_quantity(self, true_belief):
        if self.pos_price() < true_belief:
            index = 0
        elif self.neg_price() < (1.-true_belief):
            index = 1
            true_belief = 1. - true_belief # we are trying to reset the negative belief
        else:
            return np.array([0.0, 0.0])

        constant = math.exp(self.state[1-index]/self.beta)
        quant = self.beta*math.log(true_belief * constant/(1.-true_belief))-float(self.state[index])

        out = np.array([0.0, 0.0])
        out[index] = quant
        return out
