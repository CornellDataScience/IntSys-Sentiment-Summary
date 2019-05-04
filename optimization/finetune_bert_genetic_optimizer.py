from functools import partial
import optimization.genetic_optimizer as genetic_optimizer
from optimization.optimizer import Optimizer
import numpy as np


class GeneticBertOptimizer(Optimizer):


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
    '''
    '''
    The following keys are expected in config['opt_dict']:
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

         #TODO: need to update this to add a couple new arguments
    '''
    def optimize(self, X, config):
        eval_class,\
        n_elite,\
        max_sentence_ind,\
        length_range,\
        p_replace,\
        p_remove,\
        p_add,\
        max_iter,\
        prevent_dupe_sents,\
        print_iter = self._unpack_dict(config['opt_dict'],\
        "eval_class",\
        "n_elite",\
        "max_sentence_ind",\
        "length_range",\
        "p_replace",\
        "p_remove",\
        "p_add",\
        "max_iter",\
        "prevent_dupe_sents",\
        "print_iters")
        genetic_optimizer_legal_mutate_func = \
            lambda x: bert_mutation_func(x,\
                prevent_dupe_sents,
                max_sentence_ind,\
                length_range,\
                p_replace,\
                p_remove,\
                p_add)

        return genetic_optimizer.optimize(X,\
            eval_class.evaluate,\
            n_elite,\
            bert_selection_prob_func,\
            simple_bert_crossover_func,\
            genetic_optimizer_legal_mutate_func,\
            max_iter,\
            print_iter)



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
def bert_mutation_func(x, prevent_dupe_sents, max_sentence_ind, length_range, p_replace, p_remove, p_add):
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

    elif p < probs[0] + probs[1]:
        #remove
        rand_ind = np.random.randint(0, len(x))
        del x[rand_ind]

    elif p < probs[0] + probs[1] + probs[2]:
        #add
        rand_ind = np.random.randint(0, len(x))
        rand_val = np.random.randint(0, max_sentence_ind)
        x.insert(rand_ind, rand_val)

    if prevent_dupe_sents:
        unused_inds = [(-1 if i in x else i) for i in range(0, max_sentence_ind)]
        i = 0
        while i < len(unused_inds):
            if unused_inds[i] == -1:
                del unused_inds[i]
            else:
                i += 1
        i = 0
        appearance_map = {}

        for i in range(0, len(x)):
            xi_appearances = appearance_map.get(x[i])
            if xi_appearances is None:
                appearance_map[x[i]] = []
            appearance_map[x[i]].append(i)

        for dupe in appearance_map:
            dupe_appearances = appearance_map[dupe]
            if len(dupe_appearances) != 1:
                keep_appearance_ind = dupe_appearances[np.random.randint(0, len(dupe_appearances))]
                for i in dupe_appearances:
                    if i != keep_appearance_ind:
                        if len(unused_inds) == 0:
                            return x #there aren't enough candidate sentances to prevent duplicates
                        x[i] = unused_inds.pop(np.random.randint(0, len(unused_inds)))

        '''
        for i in range(len(x)):
            if len(appearance_map[x[i]]) != 1:
                keep_appearance_ind = appearance_map[x[i]][np.random.randint(0, len(appearance_map[x[i]]))]
                for repeat_ind in appearance_map[x[i]]:
                    if repeat_ind != keep_appearance_ind:
                        replace_ind = unused_inds.pop(np.random.randint(0, len(unused_inds)))
                        x[repeat_ind] = replace_ind
        '''



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

    gen_bert_opt = GeneticBertOptimizer()

    X = []
    for i in range(100):
        X.append([])
        for j in range(np.random.randint(5,15)):
            X[i].append(np.random.randint(0, 10))


    gen_bert_opt.optimize(X,\
        {
            'opt_dict': {
                'eval_class': DudEval(),
                'max_sentence_ind': 10,
                'n_elite': 10,
                'length_range': (5,15),
                'p_replace': .33,
                'p_remove': .33,
                'p_add': .33,
                'max_iter': 100,
                'print_iters': 10
            }
        })
