from market import LMSRMarket
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

print 'testing cost calculation...'



if not error:
    print('All tests passed!')

else:
    print 'Uh-oh, looks like there are some problems'
