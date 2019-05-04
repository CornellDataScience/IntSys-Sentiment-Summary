import indicoio
import pickle
from config import config
from helpers import find_clusters, sample, abstractive_clustering, cut_out_shorts
import json

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
    sentence_labels, num_clusters = find_clusters(features, config)
    return sentence_labels, num_clusters


def extractive_decode(list_of_sentences, sentence_labels, features, num_clusters, config):
    """Implements the 'encode' and 'decode' methods of the pipeline. density_parameter
        and minimum_samples are DBSCAN hyperparameters."""
        
    #sample sentences from each cluster
    candidates = sample(list_of_sentences, sentence_labels, features, num_clusters, config)
    return candidates


def abstractive_cluster(features):
    """For abstractive clustering. Returns a mean of each cluster."""
    
    sentence_labels, _ = find_clusters(features, config)
    return abstractive_clustering(features, sentence_labels)
    
if __name__ == "__main__":
    import numpy as np
    #using test data of most popular review from electronics
    #text = pickle.load(open("popular_electronics_sentences.p","rb"))
    #print("loading text")
    #text = json.load(open("data.json","r"))
    #print(np.shape(text["0"]))
    #print("loading features")
    data = pickle.load(open("sample_embeddings.p","rb"))
    data = data["B002YU83YO"]
    text = data["sentences"]
    features = data["embeddings"]
    features = np.asarray(features)
    text, features = cut_out_shorts(text, features, config)
    print(np.shape(text))
    print(np.shape(features))
    #means = abstractive_cluster(features)
    #print(np.shape(means))
    
    print("clustering...")
    sentence_labels, num_clusters = extractive_cluster(features)
    sentences = extractive_decode(text, sentence_labels, features, num_clusters, config)
    print("Number of clusters: " + str(num_clusters))
    print("Number of candidates: " + str(len(sentences)))
    
    print(sentences[::len(sentences)//num_clusters])
    
    def cluster(encodings, sentences, config):
        if True:
            sentence_labels, num_clusters = find_clusters(encodings, config)
            candidate_sentences = sample(sentences, sentence_labels, encodings, 
                                         num_clusters, config)
            return candidate_sentences
        else:
            sentence_labels, num_clusters = find_clusters(encodings, config)
            print("Number of clusters: " + str(num_clusters))
            means = []
            for cluster in set(sentence_labels):
                print("CLUSTER " + str(cluster) + "\n")
                if cluster == -1:
                    pass
                
                cluster_indices = np.where(sentence_labels == cluster)
                for i in cluster_indices[0][:10]:  
                    print(sentences[i])
                print("\n")
                cluster_core_samples = encodings[cluster_indices]
                average = np.mean(cluster_core_samples, axis = 0)
                means.append(average)
            print(len(means))
            return means
    
    #print(cluster(features, text, config))
    cluster(features, text, config)
    #print(np.shape(cluster(features, text, config)))
    #print(type(cluster(features, [], config)))
    #print(type(cluster(features, [], config)[0]))
    
"""
B000QUUFRW
B000JE7GPY
B000WL6YY8
B003ES5ZUU
B002YU83YO
B008NMCPTQ
B003LSTD38
B000WYVBR0
B001GTT0VO
B0043WJRRS
B00902SFC4
B00GTGETFG
B00007EDZG
B002TLTGM6
B0088CJT4U
"""