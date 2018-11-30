#!/usr/bin/env python

# Log Market Scoring Rule Simulation
# Rangel Milushev, Gopal Vashishtha, Tomislav Zabcic-Matic

from optparse import OptionParser
from util import mean
import copy
import logging
import random
import sys


#TODO - decide how we are tracking bids
# Do the scoring

def sim(config):
    n_agents = config.num_agents
    base_domain = [0,40,100]

    # Using the method from Hanson 2004
    agent_domains = [[0,40,100] for _ in range(n_agents)]
    true_value = random.choice([0,40,100])

    vals_to_remove = copy.deepcopy(base_domain)
    vals_to_remove.remove(true_value)

    set_1 = set(random.sample(range(n_agents), n_agents/2))
    set_2 = set()
    for i in range(n_agents):
        if i not in set_1:
            set_2.add(i)
    shuffled_agents = random.shuffle(list(range(n_agents)))

    # remove one value from half of the agents' domains
    for agent in set_1:
        agent_domains[agent].remove(vals_to_remove[0])

    for agent in set_2:
        agent_domains[agent].remove(vals_to_remove[1])

    print('agent-domains {}, true_value {}'.format(agent_domains, true_value))

    #TODO: 1) loop over rounds
    # 2) each round, each agent gets a noisy sample of the mean
    # 3) each agent then buys the security that matches its beliefs with certain probability
    # tally up wins at the end

    for t in range(config.num_rounds):
        agent_order = random.shuffle(list(range(n_agents)))
        for agent in agent_order:
            signal = random.normalvariate(true_value, agent_sigmas[agent])
            if signal > agent_values[agent]:
                buy_positive(agent)
            elif signal < agent_values[agent]:
                buy_negative(agent)

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
                      dest="num_rounds", default=48, type="int",
                      help="Set number of rounds")

    parser.add_option("--num-agents",
                      dest="num_agents", default=2, type="int",
                      help="Set number of agents")

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
