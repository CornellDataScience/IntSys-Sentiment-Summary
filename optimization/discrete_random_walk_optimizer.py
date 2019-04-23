import numpy as np
#test with levenshtein distance to reconstruct the input
'''
cand_func is a function that takes in an input candidate, x, and a
candidate function "state", S (this state need only be interpretable by
cand_func. For example, S could encode ant pheremone trails in the
case of ant colony optimization, or it could be None and not used
by cand_func at all depending on the implemtentation of cand_func).

cand_func(x, s) is expected to output cands, probs, s', where
cands is a list of candidates able to be stepped to from x,
probs is a numpy array such that probs[i] is proportional to
the probability of the random walk stepping from x into cands[i],
and s' is the state of cand_func that encodes all necessary information
for cand_func to be called once again for the next step.

X_0 is a list of initial iterates (can be size one), and the algorithm's
output will be X, s, where X is a list of final iterates, and s
is the last state returned by cand_func so that the algorithm may be
reinitialized fully treating X as X_0
'''
def optimize(X_0, cand_func, converge_crit, s = None):
    #use a queue of the best solutions found so far that gets updated after every
    #step
