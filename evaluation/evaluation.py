#TODO: write usage
import rouge

def get_scores(hyp_list, ref_list):
    '''
    obtain various rouge scores containing f, p, r
    for each corresponding entry in hyp_list (list of hypothesis entries) 
    and ref_list (list of reference entries), both which are lists of 
    string text/file paths
    '''
    assert len(hyp_list) == len(ref_list)

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=True,
                           apply_best=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

    r_scores = []
    for x in range(len(hyp_list)):
        r_score = evaluator.get_scores(hyp_list[x], ref_list[x])
        r_scores.append(r_score)

    return r_scores

def extract_rtype(r_scores, r_type):
    '''
    extract list of specified rouge type entries containing 
    f, p, r data for r_scores list 

    can extract rtypes (specify in second parameter): 
        'rouge-1',
        'rouge-2', 
        'rouge-3', 
        'rouge-4', 
        'rouge-l', 
        'rouge-w'
    '''
    assert r_type in ['rouge-1','rouge-2', 
        'rouge-3', 'rouge-4', 'rouge-l', 'rouge-w']

    return list(map(lambda c: c[r_type], r_scores))



def extract_fpr(r_type, fpr):
    '''
    extract f/p/r (specify char in second parameter) 
    and make list using each entry in r_type list
    '''
    assert fpr in ['f', 'p', 'r']
    return list(map(lambda c:c[fpr], r_type))



def get_avg(nums):
    '''
    calculate average of nums list
    '''
    assert (all(list(map(lambda c: (type(c) == float) 
                or (type(c) == int), nums))))
    return sum(nums)/len(nums)


def evaluate(hyp_list, ref_list):
    #TODO: write docstring
    print("outputting f, p, r for each rouge metric...")

    scores = get_scores(hyp_list, ref_list)
    
    r1s = extract_rtype(scores, 'rouge-1')
    print("rouge-1 metric")
    print("f: {} p: {} r: {}".format(get_avg(extract_fpr(r1s, 'f')), 
                                    get_avg(extract_fpr(r1s, 'p')), 
                                    get_avg(extract_fpr(r1s, 'r'))))
    
    r2s = extract_rtype(scores, 'rouge-2')
    print("rouge-2 metric")
    print("f: {} p: {} r: {}".format(get_avg(extract_fpr(r2s, 'f')), 
                                    get_avg(extract_fpr(r2s, 'p')), 
                                    get_avg(extract_fpr(r2s, 'r'))))
    
    r3s = extract_rtype(scores, 'rouge-3')
    print("rouge-3 metric")
    print("f: {} p: {} r: {}".format(get_avg(extract_fpr(r3s, 'f')), 
                                    get_avg(extract_fpr(r3s, 'p')), 
                                    get_avg(extract_fpr(r3s, 'r'))))
    
    r4s = extract_rtype(scores, 'rouge-4')
    print("rouge-4 metric")
    print("f: {} p: {} r: {}".format(get_avg(extract_fpr(r4s, 'f')), 
                                    get_avg(extract_fpr(r4s, 'p')), 
                                    get_avg(extract_fpr(r4s, 'r'))))
    
    rls = extract_rtype(scores, 'rouge-l')
    print("rouge-l metric")
    print("f: {} p: {} r: {}".format(get_avg(extract_fpr(rls, 'f')), 
                                    get_avg(extract_fpr(rls, 'p')), 
                                    get_avg(extract_fpr(rls, 'r'))))
    
    rws = extract_rtype(scores, 'rouge-w')
    print("rouge-w metric")
    print("f: {} p: {} r: {}".format(get_avg(extract_fpr(rws, 'f')), 
                                    get_avg(extract_fpr(rws, 'p')), 
                                    get_avg(extract_fpr(rws, 'r'))))



###test###
# scores = get_scores(["the cat ate the rat", "hello hi world"], ["the fat cat ate the rat", "hello world"])
# print(scores)
# print(type(scores[0]))

# rouge_1s = extract_rtype(scores, 'rouge-1')
# print(rouge_1s)

# recalls = extract_fpr(rouge_1s, 'r')
# print(recalls)

# avg_recalls = get_avg(recalls)
# print(avg_recalls)


evaluate(["the cat ate the rat", "hello hi world"], ["the fat cat ate the rat", "hello world"])