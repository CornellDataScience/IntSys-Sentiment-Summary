from func_arg_dict import FuncArgDict
import genetic_optimizer
from optimizer import Optimizer
import numpy as np


class GeneticBertOptimizer(Optimizer):

    '''
    The following keys are expected in params_dict:
        - "eval_class": an object that implements an evaluate(self, X) method, where X is
          a list of lists of ints of sentence choices, and returns a 1d numpy array of
          fitnesses, where the ith element is the fitness of X[i].
        - "n_elite": the number of best-fitness individuals automatically placed in the
          next generation.
        - "max_sentence_ind": the number of candidate sentences from which reviews can be constructed.
        - "length_range": a tuple, where the first element is the minimum number of
          sentences that  may be used to construct a review, and the second element
          is the maximum number of sentences that may be used to construct a review.
        - "p_replace": the probability that, when mutating x, an element of
          x is replaced with a different value.
        - "p_remove": the probability that, when mutating x, an element of x
          is removed from x.
        - "p_add": the probability that, when mutating x, a random element
          is inserted in x. (probabilities can just be relative, are normalized
          anyway when used)
        - "max_iter": the number of generations to run the genetic algorithm for
        - "print_iter": how frequently (#of iterations between prints) the
          genetic algorithm prints incremental performance.
    '''
    def __init__(self, **params_dict):
        self.__eval_class,\
        self.__n_elite,\
        self.__max_sentence_ind,\
        self.__length_range,\
        self.__p_replace,\
        self.__p_remove,\
        self.__p_add,\
        self.__max_iter,\
        self.__print_iter = self._unpack_dict(params_dict,\
        "eval_class",\
        "n_elite",\
        "max_sentence_ind",\
        "length_range",\
        "p_replace",\
        "p_remove",\
        "p_add",\
        "max_iter",\
        "print_iters")
        self.__genetic_optimizer_legal_mutate_func = \
            lambda x: bert_mutation_func(x,\
                self.__max_sentence_ind,\
                self.__length_range,\
                self.__p_replace,\
                self.__p_remove,\
                self.__p_add)


    def optimize(self, X):
        return genetic_optimizer.optimize(X,\
            self.__eval_class.evaluate,\
            self.__n_elite,\
            bert_selection_prob_func,\
            simple_bert_crossover_func,\
            self.__genetic_optimizer_legal_mutate_func,\
            self.__max_iter,\
            self.__print_iter)




#TODO: Implement
def bert_fitness_func(X, ):
    return None

'''
assumes fitnesses are positive
'''
def bert_selection_prob_func(fitnesses):
    out = np.outer(fitnesses, fitnesses)
    out = np.tril(out)
    out /= np.sum(out)
    return out

def simple_bert_crossover_func(x1, x2):
    #doesn't take any care currently to ensure a sentence isn't duplicated
    crossover_point = np.random.randint(0, min(len(x1), len(x2)))
    out = []
    if np.random.rand() < 0.5:
        for i in range(0, crossover_point):
            out.append(x1[i])
        for i in range(crossover_point, len(x2)):
            out.append(x2[i])
    else:
        for i in range(0, crossover_point):
            out.append(x2[i])
        for i in range(crossover_point, len(x1)):
            out.append(x1[i])
    return out

'''
assumes p_replace + p_remove + p_add == 1
'''
def bert_mutation_func(x, max_sentence_ind, length_range, p_replace, p_remove, p_add):
    #doesn't take any care currently to ensure a sentence isn't duplicated
    assert(len(x) >= length_range[0] and len(x) <= length_range[1])

    probs = np.array([p_replace, p_remove, p_add])
    if len(x) <= length_range[0]:
        probs[1] = 0
    elif len(x) >= length_range[1]:
        probs[2] = 0
    probs /= np.sum(probs)

    p = np.random.rand()

    if p < probs[0]:
        #replace
        rand_ind = np.random.randint(0, len(x))
        rand_val = np.random.randint(0, max_sentence_ind)
        x[rand_ind] = rand_val
        return x
    elif p < probs[0] + probs[1]:
        #remove
        rand_ind = np.random.randint(0, len(x))
        del x[rand_ind]
        return x
    elif p < probs[0] + probs[1] + probs[2]:
        #add
        rand_ind = np.random.randint(0, len(x))
        rand_val = np.random.randint(0, max_sentence_ind)
        x.insert(rand_ind, rand_val)
        return x
    return x

if __name__ == "__main__":
    class DudEval:
        '''
        should be optimized at x = [5,5,5,5,5], penalizes
        exp(-sqaured distance to [5,5,5,...]) and prefers
        inputs of lesser length
        '''
        def evaluate(self, X):
            out = np.zeros(len(X))
            for i in range(out.shape[0]):
                sum = 0
                for j in range(len(X[i])):
                    sum += X[i][j]**2
                out[i] = np.exp(-sum)*(10 - (len(X[i])-5))
            return out

    #order arguments applied when instantiating is arbitrary
    gen_bert_opt = GeneticBertOptimizer(eval_class = DudEval(),\
        max_sentence_ind = 10,\
        n_elite = 10,\
        length_range = (5,15),\
        p_replace = .33,\
        p_remove = .33,\
        p_add = .33,\
        max_iter = 100,\
        print_iters = 10)

    X = []
    for i in range(100):
        X.append([])
        for j in range(np.random.randint(5,15)):
            X[i].append(np.random.randint(0, 10))

    
    gen_bert_opt.optimize(X)
