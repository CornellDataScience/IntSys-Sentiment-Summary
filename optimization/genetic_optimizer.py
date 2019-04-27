import numpy as np

'''
- fitness_func(X)[i] = the fitness of X[i]
- selection_prob_func(fitness_func(X))[i,j] = probability X[i] and X[j] produce
  offspring. Expected that all elements above the main diagonal are 0, and
  that all elements sum to 1.
  (Matrix representation allows numpy speedups)
- crossover_func(X_1, X_2)[i] = a crossed-over child of X_1[i] and X_2[i] w/o mutations
- mutation_func(X)[i] = a randomly mutated form of X[i]
'''
def optimize(X, fitness_func, selection_prob_func, n_elite, crossover_func, mutation_func, max_iter = 10000, print_iters = 100):
    def __select(select_probs, n_pairs):
        flat_inds = np.random.choice(np.arange(0, select_probs.shape[0]*select_probs.shape[1], 1), size = n_pairs, replace = False, p = select_probs.flatten())
        out = np.zeros((2, n_pairs), dtype = np.int)
        out[0] = np.floor(flat_inds / select_probs.shape[1]).astype(np.int)
        out[1] = np.mod(flat_inds, select_probs.shape[0]).astype(np.int)
        return out

    '''
    selects the n most fit indices from fitnesses
    '''
    def __select_elite(fitnesses, n):
        return np.argpartition(-fitnesses, n)[:n]


    for k in range(0, max_iter):
        fitnesses = fitness_func(X)
        if k % print_iters == 0:
            print("avg elite fitness (" + str(k) + "): " + str(np.average(fitnesses[__select_elite(fitnesses, n_elite)])))
            #print("fitnesses (" + str(k) + "): " + str(fitnesses))
            #print("X (" + str(k) + "): " + str(X))
        select_probs = selection_prob_func(fitnesses)
        X_prime = np.zeros(X.shape, dtype = X.dtype)
        X_prime[:n_elite] = X[__select_elite(fitnesses, n_elite)]
        parent_pairs = __select(select_probs, X.shape[0] - n_elite)
        parents1 = X[parent_pairs[0]]
        parents2 = X[parent_pairs[1]]
        X_prime[n_elite:] = crossover_func(parents1, parents2)
        X_prime[n_elite:] = mutation_func(X_prime[n_elite:])
        X = X_prime
    return X


if __name__ == "__main__":

    def crossover_func(X1, X2):
        assert(X1.shape == X2.shape)
        out = np.zeros(X1.shape, dtype = X1.dtype)
        crossover_points = np.random.randint(0, X1.shape[1], size = X1.shape[0])
        for i in range(out.shape[0]):
            out[i, :crossover_points[i]] = X1[i, :crossover_points[i]]
            out[i, crossover_points[i]:] = X2[i, crossover_points[i]:]
        return out

    def mutation_func(X):
        out = X.copy()
        for i in range(out.shape[0]):
            #j = np.random.randint(0, out.shape[1])
            #out[i,j] += np.random.randint(-1, 2)
            out[i] += np.random.randint(-1, 2)
        return out


    def selection_prob_func(fitnesses):
        normed_fitnesses = fitnesses/np.sum(fitnesses)
        out = np.zeros((fitnesses.shape[0], fitnesses.shape[0]))
        for i in range(1, out.shape[0]):
            for j in range(0, i - 1):
                out[i,j] = normed_fitnesses[i]*normed_fitnesses[j]
        out /= np.sum(out)
        return out


    def mixture_of_gaussians_fitness_func(centers, weights):
        def f(X):
            out = np.zeros(X.shape[0])
            for i in range(len(centers)):
                out += weights[i] * np.exp(-0.1*np.square(np.linalg.norm(X - centers[i] , axis = 1)))
            return out
        return f

    #def fitness_func(X):
    #    return np.exp(-0.01*np.square(np.linalg.norm(X, axis = 1)))

    def neighbors_func(X):
        cols = np.random.randint(0, X.shape[1], size = X.shape[0])
        out = X.copy()
        out[:,cols] += np.random.randint(-1,2,size = out.shape[0])
        return out

    X_0 = np.random.randint(-5, 6, size = (300, 25))
    n_gaussians = 5
    centers = (3*(np.random.rand(n_gaussians, X_0.shape[1])-.5)).astype(np.int).astype(np.float32)
    weights = 0.1 + 10*np.random.rand(n_gaussians)
    fitness_func = mixture_of_gaussians_fitness_func(centers, weights)
    print("fitness_func upper bound possible (a very weak upper bound): ", np.sum(weights))
    optimize(X_0, fitness_func, selection_prob_func, 5, crossover_func, mutation_func)
