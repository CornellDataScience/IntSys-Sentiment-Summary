'''
This is where we put all the pieces together.
The pipeline is as follows:
-process input reviews (cleaning + sentence segmentation)
-encode all sentences
-cluster encodings to get candidate points
-decode candidates
-optimize candidates to maximize BERT score
-evaluate'''

import pandas as pd
import gzip
import evaluation as ev

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#df = getDF('reviews_Video_Games.json.gz')

def process(reviews):
    raise NotImplementedError
    #return sentences

def encode(sentences):
    raise NotImplementedError
    #return encodings

def cluster(encodings):
    raise NotImplementedError
    #return candidate_points

def decode(candidate_points):
    raise NotImplementedError
    #return candidate_sents

def optimize(candidate_sents):
    raise NotImplementedError
    #return solution

def summarize(reviews):
    raise NotImplementedError

def evaluate(hypothesis, reference, r_type):
    '''
    r_type contains ['rouge-1','rouge-2', 
    'rouge-3', 'rouge-4', 'rouge-l', 'rouge-w']
    '''
    rouge = ev.evaluate_rouge(hypothesis, reference, r_type)
    cos_sim = ev.evaluate_embeddings(encode(hypothesis), encode(reference))

    return rouge, cos_sim
