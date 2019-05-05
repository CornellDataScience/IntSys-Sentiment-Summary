import numpy as np




def __get_most_elite_inds(fitnesses, n):
    return np.argpartition(-fitnesses, n)[:n]

def __select(select_probs, n):
    flat_inds = np.random.choice(np.arange(0, select_probs.shape[0]*select_probs.shape[1], 1), size = n, replace = False, p = select_probs.flatten())
    out = np.zeros((2, n), dtype = np.int)
    out[0] = np.floor(flat_inds / select_probs.shape[1]).astype(np.int)
    out[1] = np.mod(flat_inds, select_probs.shape[0]).astype(np.int)
    return out


'''
- X is list of initial iterates, where X[i] is the ith iterate.
  X must be compatible with fitness_func(X).
- fitness_func(X) is a numpy array where fitness_func[i] is the fitness of X[i]
- n_elite is a number >= 0 that represents the number of candidates that remain
  unchanged and are automatically included in the next generation.
- selection_prob_func(fitness_func(X)) is a numpy array where
  selection_prob_func(fitness_func(X))[i,j] is the probability X[i] and X[j]
  producing offspring. All elements at or above the main diagonal are zero,
  and the output is expected to sum to 1.
- crossover_func(x1, x2), where x1 and x2 are of the same format as the
  elements of X, returns y, where y is the offspring of x1 and x2 without
  mutations applied constructed through however crossing over x1 and x2
  is defined.
- mutation_func(x) randomly mutates x, where x is the same format as
  the elements of X.
'''
def optimize(X, fitness_func, n_elite, selection_prob_func, crossover_func, mutation_func, max_iter, print_iters = 10):
    '''fitness_func = hyper_args["fitness_func"]
    n_elite = hyper_args["n_elite"]
    selection_prob_func = hyper_args["selection_prob_func"]
    crossover_func = hyper_args["crossover_func"]
    mutation_func = hyper_args["mutation_func"]
    max_iter = hyper_args["max_iter"]
    print_iters = hyper_args["print_iters"]'''

    fitnesses = fitness_func(X)
    for k in range(0, max_iter):

        elite_inds = __get_most_elite_inds(fitnesses, n_elite)
        X_prime = []
        for elite_ind in elite_inds:
            X_prime.append(X[elite_ind])

        select_probs = selection_prob_func(fitnesses)
        X_prime_parents = __select(select_probs, len(X) - len(X_prime))
        for pair in range(X_prime_parents.shape[1]):
            x = crossover_func(X[X_prime_parents[0,pair]], X[X_prime_parents[1,pair]])
            x = mutation_func(x)
            X_prime.append(x)

        X = X_prime
        if k % print_iters == 0:
            print("elite fitness (" + str(k) + "): " + str(fitnesses[elite_inds]))

        fitnesses = fitness_func(X)

    X_fitnesses = [(X[i], fitnesses[i]) for i in range(len(fitnesses))]
    X_fitnesses = sorted(X_fitnesses, key = lambda x: -x[1])
    for i in range(len(X_fitnesses)):
        X[i] = X_fitnesses[i][0]
    return X




if __name__ == "__main__":

    def crossover_func(x1, x2):
        crossover_point = np.random.randint(0, min(len(x1), len(x2)))
        out = []
        for i in range(0, crossover_point):
            out.append(x1[i])
        for i in range(crossover_point, len(x2)):
            out.append(x2[i])
        return out

    def mutation_func(x):
        out = []
        for i in range(len(x)):
            out.append(x[i] + np.random.randint(-1, 2))
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
            out = np.zeros(len(X))
            for i in range(len(centers)):
                for j in range(len(out)):
                    out[j] += weights[i] * np.exp(-.01*np.square(np.linalg.norm(X[j] - centers[i])))
            return out
        return f


    X_0 = np.random.randint(-5, 6, size = (50, 25))
    n_gaussians = 5
    centers = (3*(np.random.rand(n_gaussians, X_0.shape[1])-.5)).astype(np.int).astype(np.float32)
    weights = 0.1 + 10*np.random.rand(n_gaussians)
    fitness_func = mixture_of_gaussians_fitness_func(centers, weights)
    print("fitness_func upper bound possible (a very weak upper bound): ", np.sum(weights))

    from func_arg_dict import FuncArgDict

    f_arg_dict = FuncArgDict(optimize, fitness_func, 5, selection_prob_func, crossover_func, mutation_func)
    f_arg_dict.call(X_0)
