#!/usr/bin/env python

# Log Market Scoring Rule Simulation
# Rangel Milushev, Gopal Vashishtha, Tomislav Zabcic-Matic

from optparse import OptionParser
from util import mean
from market import LMSRMarket, LMSRProfitMarket
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

    true_prob = config.true_prob

    base_holdings = np.array([0.,0.])

    mkt_probs = []
    mkt_revenues = []
    mkt_payoffs = []

    # Store the lower and upper bounds on "market probabilities" at each
    # time step, where market probability is the aggregate market belief that
    # an event will occur

    mkt_lower_bounds = [[] for _ in range(config.num_trials)]
    mkt_upper_bounds = [[] for _ in range(config.num_trials)]

    agent_beliefs = [[0. for _ in range(config.num_trials)] for i in range(n_agents)]
    agent_payoffs = copy.deepcopy(agent_beliefs)
    agent_utils = copy.deepcopy(agent_beliefs)

    for k in range(config.num_trials):
        agent_budgets = [float(a.budget) for a in agents]
        agent_holdings = [copy.deepcopy(base_holdings) for i in range(n_agents)]
        #agent_payoffs = [0. for i in range(n_agents)]
        #agent_utilities = [0. for i in range(n_agents)]

        if config.mkt_type == 'LMSR':
            market = LMSRMarket(state=config.state, alpha=config.alpha_lmsr, beta=config.beta_lmsr)
        else:
            market = LMSRProfitMarket(state=config.state, alpha=config.alpha_lmsr, beta=config.beta_lmsr)
        logging.debug('market is {}, base_holdings are {}'.format(market, base_holdings))
        for t in range(config.num_rounds):
            #agent_order = list(range(n_agents))
            random.shuffle(agents)
            for agent in agents:
                # draw a 1 with probability drawn from beta distribution
                drawn_value = random.betavariate(config.true_alpha, config.true_beta)
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
                    logging.info('negative price {} for trade {} agent {}'.format(price, trade, agent))

                if price < agent_budgets[agent.id]:
                    logging.debug('able to trade! executing {}'.format(trade))
                    market.trade(trade)
                    agent_holdings[agent.id] += trade
                    agent_budgets[agent.id] -= float(price)
                else:
                    logging.debug('not enough money to trade')
            mkt_lower_bounds[k].append(market.lower_bound())
            mkt_upper_bounds[k].append(market.upper_bound())

        logging.debug(market)

        # Decide on the outcome of the simulation
        if random.random() < true_prob:
            outcome = True
        else:
            outcome = False

        mkt_payoff = 0.
        for agent in agents:
            if outcome:
                payoff = agent_holdings[agent.id][0]
                agent_payoffs[agent.id][k] += payoff
            else:
                payoff = agent_holdings[agent.id][1]
                agent_payoffs[agent.id][k] += payoff
            amt_spent = config.budget-agent_budgets[agent.id]
            agent_utils[agent.id][k] += payoff-amt_spent
            mkt_payoff += payoff
            agent_beliefs[agent.id][k] = agent.cur_belief()

        mkt_probs.append(mean([mean(mkt_lower_bounds[k]), mean(mkt_upper_bounds[k])]))
        mkt_revenues.append(market.revenue)
        mkt_payoffs.append(mkt_payoff)

        logging.debug('market is {}'.format(market))

    # decide payments
    if config.output:
        print '\n\n ---------------------------'
        print 'simulation over, true probability was {}, avg market probability {} \n\n'.format(true_prob, mean(mkt_probs))
        logging.debug('market lower bounds {} \n\n market upper bounds {}\n\n'.format(mkt_lower_bounds, mkt_upper_bounds))

        for agent in agents:
            print 'agent {} avg payoff {} avg utility {} avg ending belief {}'.format(agent, mean(agent_payoffs[agent.id]), mean(agent_utils[agent.id]), mean(agent_beliefs[agent.id]))

        print 'On average over {} trials, {} rounds each, the market collected revenue {}, paid {}, achieved profit {}'.format(config.num_trials, config.num_rounds, mean(mkt_revenues), mean(mkt_payoffs), mean(mkt_revenues)-mean(mkt_payoffs))

    return [agents, true_prob, mkt_revenues, mkt_probs, mkt_lower_bounds, mkt_upper_bounds, agent_beliefs, mkt_payoffs]

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
    constructor that takes an id, a budget, a true alpha, a true beta, a
    noise value, an alpha, and a beta in that order."""
    n = len(conf.agent_class_names)
    params = zip(range(n), itertools.repeat(conf.budget), itertools.repeat(conf.true_alpha), itertools.repeat(conf.true_beta), itertools.repeat(conf.noise))
    def load(class_name, params):
        agent_class = conf.agent_classes[class_name]
        agent_mu = random.betavariate(conf.true_alpha, conf.true_beta)
        agent_alpha = agent_mu * conf.sigma
        agent_beta = conf.sigma - agent_alpha
        params += (agent_alpha, agent_beta)
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

    parser.add_option("--num_rounds",
                      dest="num_rounds", default=10, type="int",
                      help="Decide how many times agents get to bid")

    parser.add_option("--budget",
                      dest="budget", default=2., type="float",
                      help="Set agent budgets")

    parser.add_option("--seed",
                      dest="seed", default=None, type="int",
                      help="seed for random numbers")

    parser.add_option("--sigma",
                      dest="sigma", default=None, type="int",
                      help="alpha + beta for agent priors")

    parser.add_option("--noise",
                      dest="noise", default=0.0, type="float",
                      help="at noise = 0, agents always correctly interpret their signal")

    parser.add_option("--alpha_lmsr",
                      dest="alpha_lmsr", default=.05, type="float",
                      help="see page 14:10, section 3.5 in Othman")

    parser.add_option("--beta_lmsr",
                      dest="beta_lmsr", default=1.0, type="float",
                      help="Beta for normal LMSR")

    parser.add_option("--mkt_type",
                      dest="mkt_type", default='LMSR', type="string",
                      help="Choose either LMSR or LMSRProfit")

    parser.add_option("--num_trials",
                      dest="num_trials", default=10, type="int",
                      help="Decide how many times to run the market")

    parser.add_option("--initial_state",
                      dest="state", default='0.01,0.01', type="string",
                      help="How many share already sold?")

    parser.add_option("-v",
                      action="store_true", dest="output", default=True)

    parser.add_option("-q",
                      action="store_false", dest="output")

    (options, args) = parser.parse_args()

    configure_logging(options.loglevel)

    if options.seed != None:
        random.seed(options.seed)

    initial_state = options.state.split(',')
    options.state = map(float, initial_state)

    if len(args) == 0:
        # default
        agents_to_run = ['BuyOne', 'BuyOne', 'BuyOne']
    else:
        agents_to_run = parse_agents(args)

    n_agents = len(agents_to_run)
    total_draws = n_agents * options.num_rounds

    if options.sigma is None:
        options.sigma = total_draws
    # TODO - every agent gets a different prior, based on sigma

    assert(total_draws > 1)
    # parameters for true underlying probability
    options.true_alpha = float(random.randint(1, total_draws-1))
    options.true_beta = float(total_draws - options.true_alpha)
    options.true_prob = options.true_alpha/(options.true_alpha+options.true_beta)

    # Add some more config options
    options.agent_class_names = agents_to_run
    options.agent_classes = load_modules(options.agent_class_names)

    if options.output:
        logging.info("Starting simulation...")
    return sim(options)


if __name__ == "__main__":
    main(sys.argv)
