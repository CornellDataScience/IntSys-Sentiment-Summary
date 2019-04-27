import indicoio
import pickle
from helpers import cut_out_shorts, find_clusters, sample, abstractive_clustering

def extractive_encode(list_of_sentences, savePath = None):
    """Featurizes a list of sentences using the indico API"""
    print("Featurizing " + str(len(list_of_sentences)) + 
          " sentences - remember to be careful with API credits!")
    API_KEY = "Private - contact if you need it!"
    indicoio.config.api_key = API_KEY 
    sentencefeatures = indicoio.text_features(list_of_sentences, version = 2)
    if savePath != None:
        pickle.dump(sentencefeatures, open(savePath,"wb"))
    return sentencefeatures


def extractive_cluster(features):
    #loop DBSCAN, lowering threshold for density parameter until at least min_clusters
    #clusters are created, and raising if exceeding max_acceptable_clusters
    sentence_labels, num_clusters, _ = find_clusters(features)
    return sentence_labels, num_clusters


def extractive_decode(list_of_sentences, sentence_labels, features, num_clusters):
    """Implements the 'encode' and 'decode' methods of the pipeline. density_parameter
        and minimum_samples are DBSCAN hyperparameters."""
        
    #sample sentences from each cluster
    candidates = sample(list_of_sentences, sentence_labels, features, num_clusters)
    return candidates


def abstractive_cluster(features):
    """For abstractive clustering. Returns a mean of each cluster."""
    
    sentence_labels, _, _ = find_clusters(features)
    return abstractive_clustering(features, sentence_labels)
    
if __name__ == "__main__":
    import numpy as np
    #using test data of most popular review from electronics
    text = pickle.load(open("popular_electronics_sentences.p","rb"))
    features = pickle.load(open("electronics_popular_features.p","rb"))
    text, features = cut_out_shorts(text, features)
    means = abstractive_cluster(features)
    print(np.shape(means))
    
    sentence_labels, num_clusters = extractive_cluster(features)
    sentences = extractive_decode(text, sentence_labels, features, num_clusters)
    print("Number of candidates: " + str(len(sentences)))
    print(sentences)