# scm_cluster_analysis.py
# 29th April 2019
# IntSys-Summarization

from gensim import corpora

from gensim.corpora import Dictionary
from gensim.test.utils import common_texts
from gensim.similarities import SoftCosineSimilarity

import utils

# ================== CONSTANTS ==================
TOP_SIMS = 5

def create_random_corpus(sentences, labels, num_sents):
    num_clusters = len(set(labels))
    num_per_cluster = int(num_sents / num_clusters)
    rand_idxs = []

    for c in range(num_clusters):
        rand_idxs += list(np.random.choice(np.where(labels == c)[0], num_per_cluster))

    # sample remaining
    rand_idxs += list(np.random.choice(range(len(labels)), num_per_cluster)) if num_sents - (num_per_cluster * num_clusters) != 0 else []

    return [sentences[i] for i in rand_idxs]

def generate_corpus(sentences):
    documents = list(map(sentences, lambda x : [w for w in sentencs.split() if w not in utils.STOP_WORDS]))
    dictionary = corpora.Dictionary(documents)
    bow_corpus = [dictionary.doc2bow(doc) for doc in documents]

    return bow_corpus, dictionary

def get_sim_index(wv_model, bow_corpus, dictionary):
    termsim_index = WordEmbeddingSimilarityIndex(wv_model.wv)
    similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary)
    docsim_index = SoftCosineSimilarity(bow_corpus, similarity_matrix, num_best=10)

    return docsim_index

def cluster_sent_sim(sent, sim_index, dictionary):
    sims = sim_index[dictionary.doc2bow(sent.split())] #[(<doc_idx>, <score>), ...]
    total = 0
    for s in sims[:TOP_SIMS]:
        total += s[1]
    return total / TOP_SIMS