from base_agent import BaseAgent

class Truthful(BaseAgent):
    # This agent bids until instantaneous price matches its belief
    def name(self):
        return 'Truthful'
    def __repr__(self):
        # Return a string representing this agent
        return 'Truthful agent belief {} budget {}'.format(self.cur_belief(), self.budget)

    def calc_quantity(self, market):
        # Should return a numpy array [x, y] where x represents the quantity
        # of q0 we wish to buy and y represents the quantity of q1 we wish
        # to buy
        return market.calc_quantity(self.cur_belief())
