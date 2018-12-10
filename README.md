# CS136 Final Project: Simulating a Profitable LMSR

## Running the simulation
```
python simulation.py [options] PeerClass1[,cnt] PeerClass2[,cnt] ...
```
## Simulation options
+ **--loglevel** Set the logging level: 'debug' or 'info'; default is set to info
+ **--num_rounds** Decide how many times agents get to bid i.e. _--num_rounds=100_; default is set to 10
+ **--budget** Set agents' budget i.e. _--budget=5._
+ **--seed** Seed for random numbers i.e. _--seed=2_; default is _None_
+ **--sigma** Sigma = alpha + beta for agent priors; type is _int_; default is _None_
+ **--noise** At noise = 0, agents always correctly interpret their signal
+ **--alpha_lmsr** For Profitable LMSR; see page 14:10, section 3.5 in [Othman et al. 2013](https://www.cs.cmu.edu/~sandholm/liquidity-sensitive%20automated%20market%20maker.teac.pdf); The higher the alpha, the more we charge. For alpha _>1._ no trades will go through
+ **---beta_lmsr** The beta for normal LMSR; The higher the beta the more liquidity; With a higher beta the market becomes less sensitive to each trade and prices converge to _1/m_ where _m_ is the number of outcomes 
+ **--mkt_type** Choose either LMSR or LMSRProfit; LMSRProfit is the modified version
+ **--num_trials** Decide how many times to run the market i.e. _--num_trials=100_; default is set to 10
+ **--initial_state** How many shares already sold? Default is _0.01,0.01_; At _0.,0._ the LMSRProfit will break so run with value _>0._
+ **-v** Store the output value
+ **-q** Don't store output value; default is __-q__

## Some interesting trials to run

__PA__ means prediction accuracy

+ **PA/profit vs market composition** play around with the populations of agents (Truthful, BuyOne, ZeroI)
+ **PA/profit vs beta** (LMSR only) vary populations
+ **PA vs alpha** for LMSRProfit only; alpha in (0.,1.] (will fail at 1 probably, meaning no trades will occur)
+ **profit vs alpha** for LMSRProfit only; alpha in (0.,1.] 
+ **PA vs noise** both for LMSR and LMSRProfit; noise in [0.,1.]
+ **profit vs noise** both for LMSR and LMSRProfit; noise in [0.,1.]
+ **PA/profit vs various alpha and noise for different agent populations** see the above. 
+ **PA with most profitable alpha vs noise in [0, 1]** for LMSRProfit
+ **PA with most accurate alpha vs noise in [0, 1]** for LMSRProfit
+ **profit with most profitable alpha vs noise in [0, 1]** for LMSRProfit
+ **profit with most accurate alpha vs noise in [0, 1]** for LMSRProfit
+ **PA with most profitable noise vs alpha in [0, 1]** for LMSRProfit
+ **profit with most profitable noise vs alpha in [0, 1]** for LMSRProfit
+ **PA/profit vs initial conditions** both LMSR and LMSRProfit; initial conditions in (0.,100.],(0.,100.]; vary each condition individually and then both of them at the same time.  
+ **market probability over time** (use upper/lower bound); Do we converge to the true probability? Line graph with two lines.
+ **PA/profit vs budget** (keep population homogeneous)
+ **PA vs revenue** (write down the revenue/prediction accuracy as it happens)
