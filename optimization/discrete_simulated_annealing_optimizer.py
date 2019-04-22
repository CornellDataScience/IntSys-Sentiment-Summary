import numpy as np
from numpy_key_dict import NumpyKeyDict

'''
-modifies X in-place
-acceptance_prob_func takes (X_energies, X_prime_energies, k/k_max) (it combines the temperature and
 acceptance probabilities into one function -- simpler this way as nothing else uses the temperature)
'''
def optimize(X, energy_func, neighbors_func, acceptance_prob_func, k_max, energy_cache = None):
    '''
    Returns the energies for each element of X, caching energies for X's not already
    in the cache.
    '''
    def cache_calc_energies(X):
        if energy_cache is None:
            return energy_func(X)
        energies = np.zeros(len(X))
        uncached_X = []
        for i in range(len(X)):
            cached_energy_X_i = energy_cache.get(X[i])
            if cached_energy_X_i is not None:
                energies[i] = cached_energy_X_i
            else:
                energies[i] = np.nan
                uncached_X.append(X[i])


        where_nan = np.argwhere(np.isnan(energies))
        uncached_energies = energy_func(uncached_X)
        for i in range(len(where_nan)):
            X_i = X[i]
            X_i_energy = uncached_energies[i]
            energies[i] = X_i_energy
            energy_cache[X_i] = X_i_energy
        return energies

    X_energies = cache_calc_energies(X)
    for k in range(0, k_max):
        X_prime = neighbors_func(X)
        X_prime_energies = cache_calc_energies(X_prime)
        acceptance_probs = acceptance_prob_func(X_energies, X_prime_energies, float(k)/float(k_max))
        probs = np.random.rand(acceptance_probs.shape[0])
        where_accepted = np.where(acceptance_probs >= probs)
        X[where_accepted] = X_prime[where_accepted]
        X_energies[where_accepted] = X_prime_energies[where_accepted]
        if k%100 == 0:
            print("Energies at (" + str(k) + "): " + str(X_energies))
    return X


if __name__ == "__main__":

    def acceptance_prob_func(X_energies, X_prime_energies, x):
        T = np.exp(-10*(x**0.5))
        out = np.zeros(X_energies.shape[0])
        where_energy_worse = np.where(X_prime_energies > X_energies)
        where_energy_better = np.where(X_prime_energies <= X_energies)
        out[where_energy_worse] = 0.5 * T
        out[where_energy_better] = 1 - 0.5 * T
        return out

    def energy_func(X):
        return np.sum(np.square(X), axis = 1)

    def neighbors_func(X):
        return X + np.random.randint(-1, 2, size = X.shape)

    X_0 = np.random.randint(-5, 6, size = (20, 10))
    optimize(X_0, energy_func, neighbors_func, acceptance_prob_func, 100000, energy_cache = NumpyKeyDict())
