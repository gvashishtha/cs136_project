from market import LMSRMarket
import math
import numpy as np
import random

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

print 'testing cost function...'
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
assert(round(test.instant_price(0), 2) == 0.88)
assert(round(test.instant_price(1), 2) == 0.12)

print 'testing bounded loss...'
for i in range(10):
    test = LMSRMarket()
    amts = np.array([0., 0.])
    for j in range(100):
        q0_amt = random.random()
        q1_amt = random.random()
        test.trade(np.array([q0_amt, q1_amt]))
        amts[0] += q0_amt
        amts[1] += q1_amt
    outcome = random.choice([True, False])
    if outcome:
        loss = test.revenue-amts[0]
    else:
        loss = test.revenue-amts[1]
    try:
        assert(loss <= test.beta*math.log(2))
    except AssertionError:
        print 'failing bounded loss with mkt state {}, losing {}'.format(test.state, loss)
        error=True
        break

if not error:
    print('All tests passed!')

else:
    print 'Uh-oh, looks like there are some problems'
