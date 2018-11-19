#!/usr/bin/env python

# Log Market Scoring Rule Simulation
# Rangel Milushev, Gopal Vashishtha, Tomislav Zabcic-Matic

from optparse import OptionParser
from util import mean
import logging
import random
import sys


def sim(config):
    n_agents = config.num_agents

    agent_values = {}

    # sample agent values
    for i in range(n_agents):
        agent_values[i] = random.random()

    true_value = mean(agent_values.values())

    print('agent-values {}, true_value {}'.format(agent_values, true_value))

    #TODO: 1) loop over rounds
    # 2) each round, each agent gets a noisy sample of the mean
    # 3) each agent then buys one of each security with certain probability
    # tally up wins at the end

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
