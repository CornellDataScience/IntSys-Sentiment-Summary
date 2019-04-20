'''
Outputs most helpful review, summary, and evaluation 
of summary from input data path

Example usage as script:
python summarize.py reviews_CDs_and_Vinyl_5.json.gz

The pipeline is as follows:
-process input reviews (cleaning + sentence segmentation)
-encode all sentences
-cluster encodings to get candidate points
-decode candidates
-optimize candidates to maximize BERT score
-evaluate
'''

import sys
import pandas as pd
import gzip
import nltk
import evaluation as ev

def getDF(path):
    '''
    parsing method, returns pandas dataframe from given path
    
    param [path]: the review path, i.e. 'reviews_Video_Games.json.gz'
    '''
    print('<start getDF>')
    
    def parse(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    print('<finish getDF>')
    return pd.DataFrame.from_dict(df, orient='index')


#TODO: optimize, this takes too long
def process_reviews(df):
    '''
    return the list of all tokenized review sentences in corpus 
    
    param [df]: parsed pandas dataframe 
    '''
    print('<start process_reviews>')
    review_texts = list(df['reviewText']) 
    review_sents = list(map(lambda c: nltk.sent_tokenize(c), review_texts))
    rs_flatten = [item for items in review_sents for item in items]
    print('<end process_reviews>')

    return rs_flatten


def most_helpful_ind(rev_hp):
    '''
    [helper method] returns index of the most helpful rating 
    
    param [rev_hp]: panda core series, taken in from process method
    '''
    print('<start mhi>')
    occurences = list(map(lambda c: c[1], rev_hp))
    max_occurence = max(occurences)

    max_ind = 0
    max_hp = 0
    for x in range(len(rev_hp)):
        num_helpful = rev_hp[x][0]
        num_total = rev_hp[x][1]
        if num_total == 0: continue

        ratio_rating = num_helpful / num_total #TODO: change?
        occ_rating = num_total / max_occurence #TODO: change?
        
        overall_rating = (0.75*ratio_rating) + (0.25*occ_rating) #TODO: change?
        if overall_rating > max_hp: 
            max_hp = overall_rating
            max_ind = x

    print('<end mhi>')
    return max_ind


def most_helpful(df):
    '''
    returns the most helpful review in the review corpus
    
    param [df]: parsed pandas dataframe 
    '''
    print('<start mh>')
    review_hp = df['helpful']

    best_rev_ind = most_helpful_ind(review_hp)
    best_rev_txt = df.iloc[best_rev_ind]['reviewText']

    print('<end mh>')
    return best_rev_txt 


#TODO
def encode(sentences):
    '''
    param [sentences]: the list of all tokenized review sentences in corpus
    '''
    raise NotImplementedError
    #return encodings

#TODO
def cluster(encodings):
    raise NotImplementedError
    #return candidate_points

#TODO
def decode(candidate_points):
    raise NotImplementedError
    #return candidate_sents

#TODO
def optimize(candidate_sents):
    raise NotImplementedError
    #return solution


def summarize(sentences):
    '''
    param [sentences]: the list of all tokenized review sentences in corpus
    '''
    return '' #TODO: delete this line when finished implementing above functions

    encodings = encode(sentences)
    candidate_points = cluster(encodings)
    candidate_sents = decode(candidate_points)
    solution = optimize(candidate_sents)
    return solution


def evaluate(hypothesis, reference, r_type):
    '''
    return evaluation of hypothesis text (our output) compared to 
    reference text (gold standard)

    param [hypothesis]: string of summary our model outputted
    param [reference]: string of gold standard summary 
    param [r_type]: string specifying type of rouge metric we want to use,
        must be one of the following: ['rouge-1','rouge-2', 
                'rouge-3', 'rouge-4', 'rouge-l', 'rouge-w']
    '''
    rouge = ev.evaluate_rouge(hypothesis, reference, r_type)
    cos_sim = ev.evaluate_embeddings(encode(hypothesis), encode(reference))

    return rouge, cos_sim


def print_first_5(lst):
    '''
    print first 5 elements of lst

    param [lst]: very long list that would take too 
    long to print fully
    '''
    print_str = '['
    for elt in lst:
        print_str += repr(elt)
        print_str += '; \n'

    print_str += '... ]'
    return print_str

if __name__ == "__main__":
    reviews = sys.argv[1] #the review path
    print('Starting evaluation...')
    df = getDF(reviews)

    most_helpful = most_helpful(df) #the most helpful review
    print("Most helpful review:")
    print(most_helpful)

    review_sents = process_reviews(df) #list of all sentences in review corpus
    print("Processed review sentences:")
    print_first_5(review_sents)

    summary = summarize(review_sents) #summarized reviews
    print("Summary:")
    print(summary)

    evaluation = evaluate(summary, most_helpful, 'rouge-l')
    print("Evaluation of summary:")
    print("Rouge scores: {} // Cosine similarity: {}".format(evaluation[0], evaluation[1]))

