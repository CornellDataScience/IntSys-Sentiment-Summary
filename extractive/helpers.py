import numpy as np
from math import ceil
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


def cut_out_shorts(list_of_sentences, features, config):
    list_of_sentences = np.asarray(list(list_of_sentences))
    lengths = np.asarray([len(x) for x in list_of_sentences])
    accepted_indices = np.argwhere(lengths > config["min_chars_per_sentence"]).flatten()
    features = features[accepted_indices]
    list_of_sentences = list_of_sentences[accepted_indices]
    lengths = np.asarray([len(x) for x in list_of_sentences])
    accepted_indices = np.argwhere(lengths < 300).flatten()
    features = features[accepted_indices]
    list_of_sentences = list_of_sentences[accepted_indices]
    return list_of_sentences,features


def find_clusters(features, config):   
    eps = config["density_parameter"]
    min_clusters = config["min_clusters"]
    max_acceptable_clusters = config["max_acceptable_clusters"]
    minimum_samples = config["minimum_samples"]
    
    num_clusters = 0
    num_iterations = 0
    while num_clusters < min_clusters - num_iterations or num_clusters > max_acceptable_clusters:
        num_iterations += 1
        
        clusterer = DBSCAN(eps = eps, min_samples = minimum_samples, metric = "l2")
        sentence_labels = clusterer.fit_predict(features)
        
        if num_clusters < min_clusters:
            if max(set(sentence_labels), key=list(sentence_labels).count) == -1:
                minimum_samples = max(2, minimum_samples - 1)
                eps /= 0.85
            else:
                eps *= .6
        elif num_clusters > max_acceptable_clusters:
            minimum_samples = minimum_samples + 1
            eps *= 0.85

        
        #subtract one since DBSCAN considers non-clustered sentences to be in one large cluster
        num_clusters = len(set(sentence_labels)) - 1
        print("Iteration:" + str(num_iterations) + " Num clusters: " + str(num_clusters))
        #if changing hyperparameters seems to be failing, break out and 
        #prevent infinite loop
        if num_iterations > 25:
            print("Failed to create num_clusters between bounds " 
                  + str(min_clusters) + " and " + str(max_acceptable_clusters))
            print("Ended loop with number of clusters: " + str(num_clusters))
            break
    return sentence_labels, num_clusters


def sample(list_of_sentences, sentence_labels, features, num_clusters, config):
    candidates = []
    
    #determine how many sentences to sample from each cluster
    samples_per_cluster = ceil(config["min_num_candidates"]/num_clusters)
    
    #append samples from each cluster to candidates
    for cluster in set(sentence_labels):
        #cluster -1 is sentences that could not find a cluster, so skip
        if cluster == -1:
            continue
        cluster_indices = np.where(sentence_labels == cluster)
        cluster_core_samples = features[cluster_indices]
        average = np.mean(cluster_core_samples, axis = 0)
        distances_from_cluster = cosine_distances(features, average.reshape(1,-1))
        sample_sentence_indices = np.argsort(distances_from_cluster.flatten())[:samples_per_cluster]
        for sentence_index in sample_sentence_indices:
            candidates.append(list_of_sentences[sentence_index])
            
    return candidates


def abstractive_clustering(features, sentence_labels):
    means = []
    for cluster in set(sentence_labels):
        if cluster == -1:
            continue
        cluster_indices = np.where(sentence_labels == cluster)
        cluster_core_samples = features[cluster_indices]
        average = np.mean(cluster_core_samples, axis = 0)
        means.append(average)
    return means