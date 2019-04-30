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
import scm_cluster_analysis

# ================== CONSTANTS ==================
NUM_EXAMPLES = 3
SAMPLE_PER_SECTION = 3

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

def cluster_collapse_mean(embeddings, labels, num_around_mean):
    idxs_to_keep = []

    for l in sorted(list(set(labels))):
        l_idxs = np.where(labels == l)[0]
        l_embeddings = np.mean(embeddings[l_idxs, :], axis=0)

        sorted_idxs = np.argsort(np.linalg.norm(embeddings[l_idxs, :] - l_embeddings, axis=1))
        idxs_to_keep += list(l_idxs[sorted_idxs[ : num_around_mean]])

    return idxs_to_keep

# ================== TOPIC MOD.+ DIST. SEM. ==================

# generate_cluster_idxs --> doc_sim for each cluster
def generate_cluster_idxs(sentences, labels, wv_model):
    cluster_sim_idxs = []

    for c in sorted(list(set(labels))):
        c_idxs = np.where(labels == c)[0]

        c_corpus, c_dict = scm_cluster_analysis.generate_corpus([sentences[i] for i in c_idxs])
        c_sim_idx = scm_cluster_analysis.get_sim_index(wv_model, c_corpus, c_dict)

        cluster_sim_idxs.append(c_sim_idx)

    return cluster_sim_idxs

def cluster_topic_modelling(sentences, sentence_feat, labels, label_name):
    all_topics = []

    # TODO : Have to try various topic numbers
    no_topics = 2
    no_top_words = 5

    for l in sorted(list(set(labels))):
        l_corpus = [sentences[i] for i in np.where(labels == l)[0]]
        l_vectorizer = CountVectorizer(stop_words = utils.STOP_WORDS)
        l_bow = l_vectorizer.fit_transform(l_corpus)

        l_lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=20, learning_method='online', learning_offset=5.,random_state=0).fit(l_bow)

        # dictionary of topics
        l_topics = {} # {1 : [<keyword>, <keyword>, ...], ...}
        for topic_idx, topic in enumerate(l_lda.components_):
            l_topics[topic_idx] = [l_vectorizer.get_feature_names()[i] for i in topic.argsort()[:-no_top_words - 1:-1]]

        all_topics.append(l_topics)

    return all_topics

# ================== CLUSTER COMPARISONS ==================

# PRINT
# <cluster_repr> : <num_in_labels1> --> <num_of_diff_clusters>, <num_in_highest_cluster>, <ratio>
def compare_clusters(labels1, labels2):
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

def semantic_comparison(sent1, sent2, embeddings1, embeddings2, labels1, labels2, wv_model):
    # TODO: remove test sentences? Run on random corpus
    dict1, sim_index1 = generate_dict_and_sim_for_model(sent1, labels1, wv_model)
    dict2, sim_index2 = generate_dict_and_sim_for_model(sent2, labels2, wv_model)
    print("Created dictionaries and similarity indices.")
    random_corpus1, random_corpus2 = None, None

    model_1_results = run_single_model_scm(sent1, embeddings1, labels1, labels2, sim_index1, sim_index2, dict1, dict2)
    model_2_results = run_single_model_scm(sent2, embeddings2, labels2, labels1, sim_index2, sim_index1, dict2, dict1)

    return model_1_results, model_2_results

def generate_dict_and_sim_for_model(sents, labels, wv_model):
    dict = []
    sim_index = []

    for c in sorted(list(set(labels))):
        print("Creating dict, sim for cluster_%d" % c)
        c_idxs = np.where(labels == c)[0]
        c_sents = [sents[i] for i in c_idxs]

        c_bow_corpus, c_dict = scm_cluster_analysis.generate_corpus(c_sents)
        dict.append(c_dict)
        sim_index.append(scm_cluster_analysis.get_sim_index(wv_model, c_bow_corpus, c_dict))

    return dict, sim_index

def run_single_model_scm(sent1, embeddings1, labels1, labels2, sim_index1, sim_index2, dicts1, dicts2):
    poss_labels = sorted(list(set(labels1)))

    near_sents, near_scores, alt_near_scores = [], [], []
    mid_sents, mid_scores, alt_mid_scores = [], [], []
    far_sents, far_scores, alt_far_scores = [], [], []

    for c in poss_labels:
        c_idxs = np.where(labels1 == c)[0]
        c_embeddings = embeddings1[c_idxs, :]
        c_sorted_idxs = np.argsort(np.linalg.norm(c_embeddings - np.mean(c_embeddings, axis = 0), axis = 1))

        near_idxs = c_sorted_idxs[:SAMPLE_PER_SECTION]
        mid_points = int(len(c_idxs) / 2) - int(SAMPLE_PER_SECTION / 2), int(len(c_idxs) / 2) + (SAMPLE_PER_SECTION - int(SAMPLE_PER_SECTION / 2))
        mid_idxs = c_sorted_idxs[mid_points[0] : mid_points[1]]
        far_idxs = c_sorted_idxs[-SAMPLE_PER_SECTION : ]

        c_sents, c_scores, c_alt_scores = run_single_cluster_scm(sent1, near_idxs, labels1, labels2, sim_index1, sim_index2, dicts1, dicts2)
        near_sents.append(c_sents), near_scores.append(c_scores), alt_near_scores.append(c_alt_scores)

        c_sents, c_scores, c_alt_scores = run_single_cluster_scm(sent1, mid_idxs, labels1, labels2, sim_index1, sim_index2, dicts1, dicts2)
        mid_sents.append(c_sents), mid_scores.append(c_scores), alt_mid_scores.append(c_alt_scores)
        
        c_sents, c_scores, c_alt_scores = run_single_cluster_scm(sent1, far_idxs, labels1, labels2, sim_index1, sim_index2, dicts1, dicts2)
        far_sents.append(c_sents), far_scores.append(c_scores), alt_far_scores.append(c_alt_scores)
        print("Scores for cluster_%d" % c)

    return near_sents, near_scores, alt_near_scores, mid_sents, mid_scores, alt_mid_scores, far_sents, far_scores, alt_far_scores

def run_single_cluster_scm(sentences, sent_idxs, labels1, labels2, sim_index1, sim_index2, dicts1, dicts2):
    labels1_to_use = [labels1[i] for i in sent_idxs]
    labels2_to_use = [labels2[i] for i in sent_idxs]

    sents = [sentences[i] for i in sent_idxs]
    scores = [scm_cluster_analysis.cluster_sent_sim(sents[i], sim_index1[l], dicts1[l]) for i, l in enumerate(labels1_to_use)]
    alt_scores = [scm_cluster_analysis.cluster_sent_sim(sents[i], sim_index2[l], dicts2[l]) for i, l in enumerate(labels2_to_use)]

    return tuple(sents), tuple(scores), tuple(alt_scores)

# ================== LOGGING ==================
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
# # summary stats for one cluster
# # max, avg., some examples of the distance (large, small, med)
# def leacock_chodorow_cluster(sentences):
#     #TODO
#     pass

# def leacock_chodorow_cluster(sent1, sent2):
#     #TODO
#     pass

# Want Similarity to Corpus
# - Pick sentences in the cluster (try to ignore outliers by concentrating around the mean)
#       - See similarity of sentences to that corpus vs. a random group of sentences, other corpi
# - Use Wordnet based knowledge measures as well as PMI

# Cluster Evaluation
# 1. Semantic Coherence within a cluster:
#   - Treat Cluster as Corpus (should we find a way to ignore outliers)
#   - Index a one of the sentences into it & see avergae of top-n similar documents
#   - Treat another cluster as a corpus, random group of sentneces as another corpus
#   - Ratio of the two methods
# 2. Wordnet Approaches to Semantic Coherence of a Cluster