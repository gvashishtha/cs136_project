#!/usr/bin/env python

# Log Market Scoring Rule Simulation
# Rangel Milushev, Gopal Vashishtha, Tomislav Zabcic-Matic

from optparse import OptionParser
from util import mean
from market import LMSRMarket
import copy
import itertools
import logging
import numpy as np
import random
import sys

#random.seed(2)

# Do the scoring
def sim(config):
    agents = init_agents(config)
    logging.debug(agents)
    n_agents = len(agents)

    total_draws = n_agents * config.num_rounds
    # parameters for true underlying probability
    alpha = float(random.randint(1, total_draws))
    beta = float(total_draws - alpha)
    true_prob = alpha/(alpha+beta)

    base_holdings = np.array([0.,0.])

    agent_budgets = [float(a.budget) for a in agents]
    agent_holdings = [base_holdings for i in range(n_agents)]
    agent_payoffs = [0. for i in range(n_agents)]

    market = LMSRMarket(state=base_holdings)

    for t in range(config.num_rounds):
        agent_order = list(range(n_agents))
        random.shuffle(agent_order)
        for agent in agents:

            # draw a 1 with probability drawn from beta distribution
            drawn_value = random.betavariate(alpha, beta)
            if random.random() < drawn_value:
                signal = 1
            else:
                signal = 0
            logging.debug('true prob is {} drawn value is {}, signal is {} agent belief is {}'.format(true_prob, drawn_value, signal, agent.cur_belief()))
            for a in agents:
                a.update_prior(signal)
            trade = agent.calc_quantity(market)
            price = market.get_price(trade)

            if price < 0:
                logging.info('negative price {} for trade {}'.format(price, trade))

            if price < agent_budgets[agent.id]:
                logging.debug('able to trade! executing {}'.format(trade))
                market.trade(trade)
                agent_holdings[agent.id] += trade
                agent_budgets[agent.id] -= float(price)
            else:
                logging.debug('not enough money to trade')

    # Decide on the outcome of the simulation
    if random.random() < true_prob:
        outcome = True
    else:
        outcome = False

    # decide payments
    print('\n\n ---------------------------')
    print('simulation over, true probability was {}, outcome was {}, agent holdings {} agents\' remaining budget {}\n\n'.format(true_prob, outcome, agent_holdings, agent_budgets))
    for agent in agents:
        if outcome:
            agent_payoffs[agent.id] = agent_holdings[agent.id][0]
        else:
            agent_payoffs[agent.id] = agent_holdings[agent.id][1]
        print 'agent {} belief {} received payoff {}'.format(agent.id, agent.cur_belief(), agent_payoffs[agent.id])

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


def init_agents(conf):
    """Each agent class must be already loaded, and have a
    constructor that takes an id, a value, and a budget, in that order."""
    n = len(conf.agent_class_names)
    params = zip(range(n), itertools.repeat(conf.budget))
    def load(class_name, params):
        agent_class = conf.agent_classes[class_name]
        return agent_class(*params)

    return map(load, conf.agent_class_names, params)

def load_modules(agent_classes):
    """Each agent class must be in module class_name.lower().
    Returns a dictionary class_name->class"""

    def load(class_name):
        module_name = class_name.lower()  # by convention / fiat
        module = __import__(module_name)
        agent_class = module.__dict__[class_name]
        return (class_name, agent_class)

    return dict(map(load, agent_classes))

def parse_agents(args):
    """
    Each element is a class name like "Peer", with an optional
    count appended after a comma.  So either "Peer", or "Peer,3".
    Returns an array with a list of class names, each repeated the
    specified number of times.
    """
    ans = []
    for c in args:
        s = c.split(',')
        if len(s) == 1:
            ans.extend(s)
        elif len(s) == 2:
            name, count = s
            ans.extend([name]*int(count))
        else:
            raise ValueError("Bad argument: %s\n" % c)
    return ans

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

    if len(args) == 0:
        # default
        agents_to_run = ['BuyOne', 'BuyOne', 'BuyOne']
    else:
        agents_to_run = parse_agents(args)

    # Add some more config options
    options.agent_class_names = agents_to_run
    options.agent_classes = load_modules(options.agent_class_names)

    logging.info("Starting simulation...")
    sim(options)


if __name__ == "__main__":
    main(sys.argv)
