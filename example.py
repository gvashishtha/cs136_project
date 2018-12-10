import sys
from simulation import main
import logging
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as sp
from textwrap import wrap

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# sys.argv = [
#     'simulation.py',
#     '--initial_state=0.1,0.1',
#     '--num_rounds=3',
#     '--num_trials=2',
#     '--budget=10.',
#     '--mkt_type=LMSR',
#     '-q',
#     'BuyOne,3',
#     'Truthful,3',
# ]
# output = main(sys.argv)
#
# agents = output[0]
# true_prob = output[1]
# mkt_revenues = output[2]
# mkt_probs = output[3]
# mkt_lower_bounds = output[4]
# mkt_upper_bounds = output[5]
# agent_beliefs = output[6]
# mkt_payoffs = output[7]
#
# print 'agents in the market are {}'.format(agents)
# for agent in agents:
#     print('access agent IDs like this: {} (useful for indexing into agent_beliefs, below)'.format(agent.id))
# print 'true probability of occurrence is {}'.format(true_prob)
# print 'mkt revenues, one for each of num_trials {}'.format(mkt_revenues)
# print 'mkt payoffs, total amount paid to all agents, one for each trial {}'.format(mkt_payoffs)
# print 'mkt_probs, average market belief, one for reach of num_trials {}'.format(mkt_probs)
# print 'mkt_lower_bounds, consists of k lists, each of length l, showing lower bound on mkt belief at each of l time steps for each of k trials {}'.format(mkt_lower_bounds)
# print 'mkt_upper_bounds, defined analogously to mkt_lower_bounds {}'.format(mkt_upper_bounds)
# print 'agent_beliefs, consisting of n lists of k beliefs, where n is number of agents and k is number of trials, showing each agent\'s belief for each trial {}'.format(agent_beliefs)


sys.argv = [
    'simulation.py',
    '--initial_state=0.1,0.1',
    '--num_rounds=50',
    '--num_trials=50',
    '--budget=10.',
    '--mkt_type=LMSRProfit',
    '--alpha_lmsr=.05',
    '-q',
    'BuyOne,3',
    'ZeroI,3',
]
output = main(sys.argv)

agents = output[0]
true_prob = output[1]
mkt_revenues = output[2]
mkt_probs = output[3]
mkt_lower_bounds = output[4]
mkt_upper_bounds = output[5]
agent_beliefs = output[6]
mkt_payoffs = output[7]

def compute(mkt_bounds):
    bounds = np.array([0.]*len(mkt_bounds[0]))
    error_bars = [[] for i in range(len(mkt_bounds[0]))]
    for b in mkt_bounds:
        bounds += np.array(b)
        for t in range(len(mkt_bounds[0])): # append each time to correct list
            error_bars[t].append(b[t])
    logging.debug('error_bars before map {}'.format(len(error_bars)))
    logging.debug('bounds before map {}'.format(len(bounds)))
    error_bars = map(lambda x: sp.sem(x), error_bars)
    #print('erroR_bars is {}'.format(error_bars))

    bounds = map(lambda x: x/len(mkt_bounds), bounds) # compute average
    #print('bounds is {}'.format(bounds))
    # df = pd.Series(list(mkt_sums), index=range(len(mkt_lower_bounds[0])))
    # df.plot('ro')
    x=list(range(0, len(mkt_bounds)))
    return(x, bounds, error_bars)

(x1, b1, e1) = compute(mkt_lower_bounds)
(x2, b2, e2) = compute(mkt_upper_bounds)
logging.debug('x1 is {} b1 is {} e1 is {}'.format(len(x1), len(b1), len(e1)))

plt.title('\n'.join(wrap('Upper and lower bounds on market probability in modified LMSR. alpha = .05, agents = 3 BO, 3 ZeroI')))
plt.errorbar(x=x1,y=b1, yerr=e1, xerr=None, fmt='r-', ecolor='r',label='lower bounds')
plt.errorbar(x=x2,y=b2, yerr=e2, xerr=None, fmt='b-', ecolor='b', label='upper bounds')
plt.hlines(y=true_prob, xmin=0, xmax =len(mkt_lower_bounds),label = 'true probability')
plt.legend()
# plt.plot(mkt_lower_bounds[0], 'ro')
plt.show()
plt.save('bounds.jpg')
