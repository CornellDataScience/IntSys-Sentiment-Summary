import numpy as np

'''
See: https://en.wikipedia.org/wiki/Stochastic_tunneling, which can be performed
by correct choice of f and accept_prob_func

f(X, s), where X is a matrix whose rows are inputs, returns y, s',
where y[i] = f(X[i], s[i]), and s'[i] is the "state" of f (s can be None
if not needed by f) given for the f evaluating on the X[i]. If S is None,
f then f treates it as if it were its first evaluation

neighbors_func(X) returns a list of neighbors of X such that the ith
element of the output is a random neighbor of X[i]

returns S so that the optimization can be restarted later
from the output X and that S without issue
'''
def optimize(X, f, neighbors_func, accept_prob_func, k_max, S_0 = None):
    f_X, S = f(X, S_0)
    for k in range(0, k_max):
        X_prime = neighbors_func(X)
        f_X_prime, S_prime = f(X_prime, S)
        X_prime_accept_probs = accept_prob_func(f_X, f_X_prime)
        probs = np.random.rand(X_prime_accept_probs.shape[0])
        where_accepted = np.where(probs <= X_prime_accept_probs)

        X[where_accepted] = X_prime[where_accepted]
        if S is not None and S_prime is not None:
            S[where_accepted] = S_prime[where_accepted]
        f_X[where_accepted] = f_X_prime[where_accepted]
        if k % 100 == 0:
            print("f_X on iteration " + str(k) + ": " + str(f_X))
    return X, S


if __name__ == "__main__":
    '''
    Below implements vanilla Metropolis Hastings
    (Described in "Idea" section here:
    https://en.wikipedia.org/wiki/Stochastic_tunneling)
    to minimize f(X)
    '''
    BETA = 10.0
    def f(X, S):
        return np.sum(np.square(X), axis = 1).astype(np.float32), None

    def neighbors_func(X):
        cols = np.random.randint(0, X.shape[1], size = X.shape[0])
        out = X.copy()
        out[:,cols] += np.random.randint(-1,2,size = out.shape[0])
        return out

    def accept_prob_func(f_X, f_X_prime):
        out = np.exp(-BETA * (f_X_prime - f_X))
        out[np.where(out > 1)] = 1
        return out

    X = np.random.randint(-20, 21, size = (5, 100))
    optimize(X, f, neighbors_func, accept_prob_func, 100000, S_0 = None)
