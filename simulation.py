#!/usr/bin/env python

# Log Market Scoring Rule Simulation
# Rangel Milushev, Gopal Vashishtha, Tomislav Zabcic-Matic

from optparse import OptionParser
from util import mean
from market import LMSRMarket
import copy
import logging
import numpy as np
import random
import sys

#random.seed(2)

# Do the scoring
def sim(config):
    n_agents = config.num_agents

    # Using the method from Hanson 2004
    base_domain = [0.,0.4,1.0]
    agent_holdings = [np.array([0.,0.]) for i in range(n_agents)]
    agent_budgets = [config.budget for i in range(n_agents)]
    agent_payoffs = [0 for i in range(n_agents)]
    agent_domains = [copy.deepcopy(base_domain) for _ in range(n_agents)]

    true_value = random.choice(base_domain)

    vals_to_remove = copy.deepcopy(base_domain)
    vals_to_remove.remove(true_value)

    # For half the agents, reveal one incorrect belief.
    # For the other half, remove a different incorrect belief.
    set_1 = set(random.sample(range(n_agents), n_agents/2))
    set_2 = set()

    for i in range(n_agents):
        if i not in set_1:
            set_2.add(i)
    shuffled_agents = random.shuffle(list(range(n_agents)))

    for agent in set_1:
        agent_domains[agent].remove(vals_to_remove[0])

    for agent in set_2:
        agent_domains[agent].remove(vals_to_remove[1])

    logging.info('agent domains {}, true_value {}'.format(agent_domains, true_value))

    # To look into - maybe we should incorporate expected profit into decision about trading?

    market = LMSRMarket()

    for t in range(config.num_rounds):
        agent_order = list(range(n_agents))
        random.shuffle(agent_order)
        for agent in agent_order:
            signal = mean(agent_domains[agent])
            trade = market.calc_quantity(signal)
            price = market.get_price(trade)

            if trade[0] == 0.0: # we are trying to buy against the outcome
                exp_profit = (1. - signal) - price
            elif trade[1] == 0.0: # we are buying for the outcome
                exp_profit = signal - price

            logging.debug('signal {} market state {} exp_profit from trading {}'.format(signal, market.state, exp_profit))

            if price < agent_budgets[agent]:
                logging.debug('able to trade! executing {}'.format(trade))
                market.trade(trade)
                agent_holdings[agent] += trade
                agent_budgets[agent] -= price
            else:
                logging.debug('not enough money to trade')


    # decide payments
    print('\n\n ---------------------------')
    print('simulation over, true value was {}, agent holdings {} agents\' remaining budget {}\n\n'.format(true_value, agent_holdings, agent_budgets))
    for agent in range(n_agents):
        agent_payoffs[agent] += true_value*agent_holdings[agent][0] + (1.-true_value)*agent_holdings[agent][1]
        print('agent {} belief {} received payoff {}'.format(agent, mean(agent_domains[agent]), agent_payoffs[agent]))

    print('market collected revenue {}, paid {}, profit {}'.format(market.revenue, sum(agent_payoffs), market.revenue-sum(agent_payoffs)))

class Params:
    def __init__(self):
        self._init_keys = set(self.__dict__.keys())

    def add(self, k, v):
        self.__dict__[k] = v

    def __repr__(self):
        return "; ".join("%s=%s" % (k, str(self.__dict__[k]))
                         for k in self.__dict__.keys() if k not in self._init_keys)

def configure_logging(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    root_logger = logging.getLogger('')
    strm_out = logging.StreamHandler(sys.__stdout__)
    strm_out.setFormatter(logging.Formatter('%(message)s'))
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(strm_out)

def main(args):

    usage_msg = "Usage:  %prog [options] PeerClass1[,cnt] PeerClass2[,cnt2] ..."
    parser = OptionParser(usage=usage_msg)

    def usage(msg):
        print "Error: %s\n" % msg
        parser.print_help()
        sys.exit()

    parser.add_option("--loglevel",
                      dest="loglevel", default="info",
                      help="Set the logging level: 'debug' or 'info'")

    parser.add_option("--num-rounds",
                      dest="num_rounds", default=10, type="int",
                      help="Set number of rounds")

    parser.add_option("--num-agents",
                      dest="num_agents", default=2, type="int",
                      help="Set number of agents")

    parser.add_option("--budget",
                      dest="budget", default=2., type="float",
                      help="Set agent budgets")

    parser.add_option("--seed",
                      dest="seed", default=None, type="int",
                      help="seed for random numbers")


    (options, args) = parser.parse_args()

    configure_logging(options.loglevel)

    if options.seed != None:
        random.seed(options.seed)

    logging.info("Starting simulation...")
    sim(options)


if __name__ == "__main__":
    main(sys.argv)
