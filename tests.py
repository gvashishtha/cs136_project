from market import LMSRMarket, LMSRProfitMarket
import logging
import math
import numpy as np
import random
import sys
from simulation import main

test = LMSRMarket()
error = False

print 'testing quantity calculation...'
for i in range(100):
    belief = random.random()
    cur_state = test.state
    test.trade(test.calc_quantity(belief))
    try:
        assert(abs(test.pos_price() - belief) <= 0.001)
    except AssertionError:
        print 'failing test with belief {} market state {}'.format(belief, cur_state)
        error = True
        break

print 'testing price function...'
test = LMSRMarket(beta=1.0)
test.state = np.array([0.,0.])

def price_tester(trade, price):
    try:
        assert(round(test.get_price(trade), 2) == price)
        test.trade(trade)
    except AssertionError:
        print('failing test, real price is {}'.format(test.get_price(trade)))
        error = True

trades = [[1.,0.], [2,0.],[0.,1.]]
trades = map(np.array, trades)
prices = [0.62, 1.74, 0.08]

for t,p in zip(trades, prices):
    price_tester(t, p)

print 'testing instant price function ...'
assert(round(test.instant_price(0), 2) == 0.88)
assert(round(test.instant_price(1), 2) == 0.12)

print 'testing revenue function ...'
try:
    assert(abs(round(test.revenue, 2) - sum(prices)) <= 0.05)
    #print('cost final - cost initial is {} '.format(test.get_cost(test.state)-test.get_cost(np.array([0.,0.]))))
except AssertionError:
    print('failing revenue test, revenue {} prices sum {}'.format(test.revenue, sum(prices)))
    error = True

print 'testing bounded loss...'
def bounded_loss_test(mkt_type):
    global error
    for i in range(10):
        test3 = mkt_type(alpha=0.05, beta = random.choice(range(1,30)))
        initial_cost = test3.get_cost(test3.state)
        amts = np.array([0., 0.])
        for j in range(100):
            #print(test3)
            q0_amt = random.random()
            q1_amt = random.random()
            old_revenue = test3.revenue
            test3.trade(np.array([q0_amt, q1_amt]))
            expected_revenue = test3.get_cost(test3.state)-initial_cost
            try:
                assert(abs(test3.revenue - expected_revenue) < 0.05)
            except AssertionError:
                trade = np.array([q0_amt, q1_amt])
                old_state = test3.state - trade
                logging.error('revenue calc failing with mkt state {}, prev_state is {}, get_price {} old revenue {} new rev {} expected rev {}'.format(test3.state, old_state, test3.get_price(trade), old_revenue, test3.revenue, expected_revenue))
                error=True
                break
            amts[0] += q0_amt
            amts[1] += q1_amt
        # Decide on outcome and calculate the loss to the market maker
        outcome = random.choice([True, False])
        if outcome:
            loss = amts[0]-test3.revenue
        else:
            loss = amts[1]-test3.revenue

        # Ensure the loss is bounded by the expected bound
        try:
            bound = test3.beta*math.log(2)  # equation 18.9 in book
            if isinstance(test3, LMSRProfitMarket):
                bound = initial_cost # page 14:19 of Othman
            assert(loss <= bound)
        except AssertionError:
            final_cost = test3.get_cost(test3.state)
            initial_cost = test3.get_cost(np.array([0.,0.]))
            logging.error('failing bounded loss with mkt state {}, revenue {}, losing {} revenue should be {} bound {}'.format(test3.state, test3.revenue, loss, final_cost-initial_cost, bound))
            error=True
            break
bounded_loss_test(LMSRMarket)
bounded_loss_test(LMSRProfitMarket)

print 'testing all agents can be initialized'
sys.argv = ["simulation.py", "ZeroI,1", "Truthful,1", "BuyOne,1", '-q']
main(sys.argv)

sys.argv = ["simulation.py", "ZeroI,1", "Truthful,1", "BuyOne,1", "--mkt_type=LMSRMoney", '-q']
main(sys.argv)

print 'testing that posteriors update correctly'
def belief_tester():
    global error
    output = main(sys.argv)
    agents = output[0]
    true_prob = output[1]
    for agent in agents:
        try:
            assert(abs(agent.cur_belief()-true_prob) <= 0.05)
            logging.debug('agent {} true prob {} belief {}'.format(agent, true_prob, agent.cur_belief()))
        except AssertionError:
            logging.error('agent belief issue with {}. Exiting'.format(agent))
            error=True
            break

sys.argv = ['simulation.py', 'BuyOne,3', 'Truthful,3', '--num_rounds=100', '--mkt_type=LMSR', '-q']
belief_tester()
sys.argv = ['simulation.py', 'BuyOne,3', 'Truthful,3', '--num_rounds=100', '--mkt_type=LMSRMoney', '-q']
belief_tester()

print ('testing LMSR Profit beta calculations...')

test = LMSRProfitMarket(state=np.array([0.1,0.1]), alpha=0.5)
assert(test.calc_beta(test.state)==0.1)
try:
    assert(round(test.get_cost(test.state), 3)==0.169)
except AssertionError:
    logging.error('test state is {} beta is {} cost is {}'.format(test.state, test.beta, round(test.get_cost(test.state),3)))
    error=True

print('testing LMSR prices...')

def price_tester(trade, price):
    try:
        assert(round(test.get_price(trade), 3) == price)
        test.trade(trade)
    except AssertionError:
        print('failing test, real price is {}'.format(test.get_price(trade)))
        error = True

trades = [[1.,0.], [2,0.],[0.,1.]]
trades = map(np.array, trades)
prices = [1.034, 2.124, 0.457]

for t,p in zip(trades, prices):
    price_tester(t, p)

if not error:
    print('All tests passed!')

else:
    print 'Uh-oh, looks like there are some problems'
