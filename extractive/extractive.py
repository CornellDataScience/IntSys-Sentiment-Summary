import indicoio
import pickle
from helpers import cut_out_shorts, find_clusters, sample, abstractive_cluster

def extractive_encode(list_of_sentences, savePath = None):
    """Featurizes a list of sentences using the indico API"""
    print("Featuring " + str(len(list_of_sentences)) + 
          " sentences - remember to be careful with API credits!")
    API_KEY = "Private - contact if you need it!"
    indicoio.config.api_key = API_KEY
    sentencefeatures = indicoio.text_features(list_of_sentences, version = 2)
    if savePath != None:
        pickle.dump(sentencefeatures, open(savePath,"wb"))
    return sentencefeatures

def extractive_decode(list_of_sentences, features = None):
    """Implements the 'encode' and 'decode' methods of the pipeline. density_parameter
        and minimum_samples are DBSCAN hyperparameters."""
    
    #featurize if needed
    if features is None:
        features = extractive_encode(list_of_sentences)
    
    #cut out very short sentences
    list_of_sentences, features = cut_out_shorts(list_of_sentences, features) 

    #loop DBSCAN, lowering threshold for density parameter until at least min_clusters
    #clusters are created, and raising if exceeding max_acceptable_clusters
    sentence_labels, num_clusters, eps = find_clusters(features)

    #sample sentences from each cluster
    candidates = sample(list_of_sentences, sentence_labels, features, num_clusters)
    
    print("Number of samples generated: " + str(len(candidates)))
    print("Number of clusters: " + str(num_clusters))
    print("Final density parameter: " + str(eps))
    return candidates


def cluster(features):
    """For abstractive clustering. Returns a mean of each cluster."""
    
    sentence_labels, _, _ = find_clusters(features)
    return abstractive_cluster(features, sentence_labels)
    
if __name__ == "__main__":
    import numpy as np
    #using test data of most popular review from electronics
    text = pickle.load(open("popular_electronics_sentences.p","rb"))
    features = pickle.load(open("electronics_popular_features.p","rb"))
    text, features = cut_out_shorts(text, features)
    means = cluster(features)
    print(np.shape(means))
    sentences = extractive_decode(text, features)
    print(sentences)