'''
This is where we put all the pieces together.
The pipeline is as follows:
-process input reviews (cleaning + sentence segmentation)
-encode all sentences
-cluster encodings to get candidate points
-decode candidates
-optimize candidates to maximize BERT score
-evaluate'''

def process(reviews):
    raise NotImplementedError
    return sentences

def encode(sentences):
    raise NotImplementedError
    return encodings

def cluster(encodings):
    raise NotImplementedError
    return candidate_points

def decode(candidate_points):
    raise NotImplementedError
    return candidate_sents

def optimize(candidate_sents):
    raise NotImplementedError
    return solution

def evaluate(solution):
    raise NotImplementedError
    return metrics

def summarize(reviews):
    raise NotImplementedError