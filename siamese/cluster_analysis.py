# siamese_analysis.py
# 24th April 2019
# IntSys-Summarization

import numpy as np

from nltk.corpus import wordnet as wn
import spacy

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import utils

# ================== CONSTANTS ==================
NUM_EXAMPLES = 3

# ================== CLUSTER CREATION ==================

# Analyze Siamese Network
def kmeans_labels(embeddings, num_clusters):
    return KMeans(n_clusters=num_clusters, 
                  random_state=0).fit(embeddings).labels_

def agglomerative_labels(embeddings, num_clusters):
    return AgglomerativeClustering(n_clusters=num_clusters).fit(embeddings).labels_

# ================== CLUSTER STATISTICS ==================

# score_func : fn(feat1, feat2)
# sent_features must be a LIST of features
def compile_cluster_stats(name, sentences, sent_features, labels, score_func):
    cluster_stats = []

    for l in sorted(list(set(labels))):
        feat_idxs = np.where(labels == l)[0] #sorted order

        score_idxs = [] #[(i,j), ...]
        scores = []

        for i, fi in enumerate(feat_idxs):
            for j, fj in enumerate(feat_idxs[i + 1:]):
                feat_i = sent_features[fi]
                feat_j = sent_features[fj]

                score_ij = score_func(feat_i, feat_j)

                scores.append(score_ij)
                score_idxs.append((fi, fj))

        scores = np.array(scores)
        sorted_idxs = np.argsort(scores)

        cluster_stats.append(ClusterStats(name, *get_all_examples(sentences, scores,
                                                                  score_idxs, sorted_idxs)))
    # TODO: remove scores, scores_idxs
    return cluster_stats

# post stem, lemmatizing
def cluster_jaccard_statistics(sentences, sentences_sets, labels, label_name):
    #TODO (how to handle division by zero)
    jaccard_comp = lambda s1, s2 : len(s1.intersection(s2)) / len(s1.union(s2)) if len(s1.union(s2)) != 0 else np.inf
    jaccard_cluster_stats = compile_cluster_stats("%s : jaccard" % label_name, 
                                                  sentences, sentences_sets, 
                                                  labels, jaccard_comp)
    return jaccard_cluster_stats

def cluster_avg_vector_statistics(sentences, sentences_docs, labels, label_name):
    vector_comp = lambda d1, d2 : d1.similarity(d2)
    vec_cluster_stats = compile_cluster_stats("%s : avg_vector" % label_name, 
                                              sentences, sentences_docs, 
                                              labels, vector_comp)
    return vec_cluster_stats

def cluster_wordnet_statistics():
    #TODO
    pass

def cluster_topic_modelling(sentences, sentence_feat, labels, label_name):
    all_topics = []

    # TODO : Have to try various topic numbers
    no_topics = 4
    no_top_words = 7

    for l in sorted(list(set(labels))):
        l_corpus = [sentences[i] for i in np.where(labels == l)[0]]
        l_vectorizer = CountVectorizer(stop_words = utils.STOP_WORDS)
        l_bow = l_vectorizer.fit_transform(l_corpus)

        l_lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(l_bow)

        # dictionary of topics
        l_topics = {} # {1 : [<keyword>, <keyword>, ...], ...}
        for topic_idx, topic in enumerate(l_lda.components_):
            l_topics[topic_idx] = [l_vectorizer.get_feature_names()[i] for i in topic.argsort()[:-no_top_words - 1:-1]]

        all_topics.append(l_topics)

    return all_topics

# sents : ["<sent1>", ...]
# scores :[<score_sent_ij>, ..]
# score_idxs : [(i, j), ..]
# sorted_idxs : the order in which scores (and score_idxs) should be sorted
def get_all_examples(sents, scores, score_idxs, sorted_idxs):
    min_val = scores[sorted_idxs[0]]
    min_ex = sents[score_idxs[sorted_idxs[0]][0]], sents[score_idxs[sorted_idxs[0]][1]]

    max_val = scores[sorted_idxs[-1]]
    max_ex = sents[score_idxs[sorted_idxs[-1]][0]], sents[score_idxs[sorted_idxs[-1]][1]]

    avg_val, near_avg = closest_to_avg(score_idxs, scores)
    avg_ex = (sents[near_avg[0]], sents[near_avg[1]])

    var = np.var(scores)

    # collect examples
    mid_idx = int(len(sorted_idxs) / 2)
    example_idxs = (list(sorted_idxs[1 : 1 + int(NUM_EXAMPLES / 3)]) + 
                    list(sorted_idxs[mid_idx - 1 : mid_idx + 2]) + 
                    list(sorted_idxs[-1 - int(NUM_EXAMPLES / 3) :-1]))

    examples = []
    for i in example_idxs:
        #examples.append((sents[score_idxs[i][0]], sents[score_idxs[i][1]], scores[sorted_idxs[i]]))
        examples.append((sents[score_idxs[sorted_idxs[i]][0]], sents[score_idxs[sorted_idxs[i]][1]], scores[sorted_idxs[i]]))


    return avg_val, avg_ex, min_val, min_ex, max_val, max_ex, var, examples

def closest_to_avg(score_idxs, scores):
    avg_val = np.mean(scores)
    dist_to_avg = scores - avg_val

    min_idx = np.argmin(dist_to_avg)
    return avg_val, score_idxs[min_idx]

# ================== CLUSTER COMPARISONS ==================

# PRINT
# <cluster_repr> : <num_in_labels1> --> <num_of_diff_clusters>, <num_in_highest_cluster>, <ratio>
def cluster_comparison(labels1, labels2):
    labels = set(labels1)

    print("How much Clusters Shift Around: ")
    print("<Cluster Repr.> : <# in Cluster> ----> <# of Clusters Repr.>, <# in Highest Repr. Cluster>, <Ratio>")

    for l in labels:
        l_idxs = np.where(labels1 == l)[0]

        labels2_clusters = labels2[l_idxs]
        cluster_counts = np.bincount(labels2_clusters)
        max_idx = np.argmax(cluster_counts)

        print("%d : %d ----> %d, %d, %f" % (l_idxs[0], len(l_idxs), 
                                            len(set(labels2_clusters)), cluster_counts[max_idx], 
                                            cluster_counts[max_idx] / len(labels2_clusters)))

class ClusterStats(object):

    # *_ex : (sent1, sent2)S
    # examples : [(sent1, sent2, score), ...]
    def __init__(self, name, avg_val, avg_ex, min_val, min_ex, max_val, max_ex, var, examples):

        self.name = name
        self.avg_val = avg_val
        self.avg_ex = avg_ex 
        self.min_val = min_val
        self.min_ex = min_ex
        self.max_val = max_val
        self.max_ex = max_ex
        self.var = var
        self.examples = examples

    def __repr__(self):
        return "%s: avg= %f, min= %f, max= %f, var= %f" % (self.name, self.avg_val, 
                                                           self.min_val, self.max_val, 
                                                           self.var)

#########
# # more like qualifying semantic relatedness

# # use of Jaccard Similarity (for 1-token overlap)
# # labels : n vector
# def jaccard_cluster(sentences, sentences_set, labels):
#     cluster_stats = []

#     for l in sorted(list(set(labels))):
#         set_idxs = np.where(labels == l)[0] #sorted order

#         jaccard_idxs = [] #[(i,j), ...]
#         jaccard_scores = []

#         for i, si in enumerate(set_idxs):
#             for j, sj in enumerate(set_idxs[i + 1:]):
#                 set_i = sentences_set[si]
#                 set_j = sentences_set[sj]

#                 jaccard_ij = set_i.intersection(set_j) / set_i.union(set_j)

#                 jaccard_scores.append(jaccard_ij)
#                 jaccard_idxs.append((si, sj))

#         jaccard_scores = np.array(jaccard_scores)
#         sorted_idxs = np.argsort(jaccard_scores)

#         cluster_stats.append(ClusterStats("jaccard similarity", *get_all_examples(sentences, jaccard_scores,
#                                                                                   jaccard_idxs, sorted_idxs)))

#     return cluster_stats

# # summary stats for each cluster
# def leacock_chodorow_all_clusters(sentences, labels):
#     #TODO
#     pass

# # summary stats for one cluster
# # max, avg., some examples of the distance (large, small, med)
# def leacock_chodorow_cluster(sentences):
#     #TODO
#     pass

# def leacock_chodorow_cluster(sent1, sent2):
#     #TODO
#     pass