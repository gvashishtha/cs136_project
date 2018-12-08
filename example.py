import sys
from simulation import main


sys.argv = [
    'simulation.py',
    '--initial_state=0.1,0.1',
    '--num_rounds=3',
    '--num_trials=2',
    '--budget=10.',
    '--mkt_type=LMSR',
    '-q',
    'BuyOne,3',
    'Truthful,3',
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

print 'agents in the market are {}'.format(agents)
for agent in agents:
    print('access agent IDs like this: {} (useful for indexing into agent_beliefs, below)'.format(agent.id))
print 'true probability of occurrence is {}'.format(true_prob)
print 'mkt revenues, one for each of num_trials {}'.format(mkt_revenues)
print 'mkt payoffs, total amount paid to all agents, one for each trial {}'.format(mkt_payoffs)
print 'mkt_probs, average market belief, one for reach of num_trials {}'.format(mkt_probs)
print 'mkt_lower_bounds, consists of k lists, each of length l, showing lower bound on mkt belief at each of l time steps for each of k trials {}'.format(mkt_lower_bounds)
print 'mkt_upper_bounds, defined analogously to mkt_lower_bounds {}'.format(mkt_upper_bounds)
print 'agent_beliefs, consisting of n lists of k beliefs, where n is number of agents and k is number of trials, showing each agent\'s belief for each trial {}'.format(agent_beliefs)
