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
import numpy as np
import pandas as pd
import gzip
import nltk
import pickle
import evaluation as ev
import torch
import torchtext
#import indicoio
import utils
import sys
import json
from pathlib import Path
from extractive.helpers import find_clusters, sample
from utils.dataset import Dataset
from bert_finetune.BERTEval import BERTpredictor
from autotransformer.transformer.flow import make_model
from autotransformer.summary_ae_datahandler import make_sentence_iterator, greedy_decode
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig

def load_config(config):
    """Loads Models and Data from paths listed in the initial config"""

    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.modules['dataset'] = utils.dataset

    #data = utils.dataset.Dataset().get_from_dir(Path(config['data_path']))
    with open(config['data_path'], 'r') as fp:
        data = json.load(fp)
        config['dataset'] = data

    if not config['extractive']:
        src_vocab = torch.load(Path(config['src_vocab_path']))
        trg_vocab = torch.load(Path(config['trg_vocab_path']))
        model = make_model(len(src_vocab), len(trg_vocab))
        model.load_state_dict(torch.load(Path(config['autoencoder_path'])))

        config['src_vocab'] = src_vocab
        config['trg_vocab'] = trg_vocab
        config['autoencoder'] = model

    bert_config = BertConfig(Path(config['BERT_config_path']).__str__())
    bert = BertForSequenceClassification(bert_config, num_labels=1)
    bert.load_state_dict(torch.load(Path(config['BERT_finetune_path'])))
    config['BERT_finetune_model'] = bert

    #return config


def encode(sentences, config):
    '''
    [encode sentence] returns a list of sentence encodings

    sentences: str list
    returns 2d numpy array of encodings, shape: (n_sents, dim)
    '''
    if config['extractive']:
        API_KEY = "Private - contact if you need it!"
        indicoio.config.api_key = API_KEY
        encodings = indicoio.text_features(sentences, version = 2)
        return encodings
    else:
        model = config['autoencoder']
        model.eval()
        model.cuda()

        sent_data = make_sentence_iterator(sentences, config['ae_batchsize'])
        sent_iter, SRC, BOS_WORD, EOS_WORD, BLANK_WORD, CLS_WORD = sent_data
        SRC.vocab = config['src_vocab']

        encodings = []
        for i, batch in enumerate(sent_iter):
            src = batch.src.transpose(0, 1).cuda()
            src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2).cuda()
            batch_encodings = model.encode(src, src_mask)
            for sent_encoding in batch_encodings:
                encodings.append(sent_encoding[0,:].cpu().detach().numpy())
        return np.array(encodings)


#TODO: implement
def cluster(encodings, sentences, config):
    #encodings can be list of lists or 2d numpy array; this casting is to
    #prevent list of numpy arrays, which breaks some indexing operations

    encodings = np.asarray(encodings)
    if config['extractive']:
        sentence_labels, num_clusters = find_clusters(encodings, config)
        candidate_sentences = sample(sentences, sentence_labels, encodings,
                                     num_clusters, config)
        return candidate_sentences
    else:
        sentence_labels, _ = find_clusters(encodings, config)
        means = []
        for cluster in set(sentence_labels):
            if cluster == -1:
                continue
            cluster_indices = np.where(sentence_labels == cluster)
            cluster_core_samples = encodings[cluster_indices]
            average = np.mean(cluster_core_samples, axis = 0)
            means.append(average)

        #this returns a list of numpy arrays
        return means


#TODO: implement
def decode(candidate_points, config):
    if config['extractive']:
        return candidate_points
    else:
        return greedy_decode(config['autoencoder'], candidate_points, config['trg_vocab'])

#TODO: implement
def optimize(candidate_sents, config):
    bert = BERTpredictor(config, candidate_sents)
    best_sents = peter_optimizer(bert, candidate_sents, config)[0]
    review = ' '.join([bert.sentences[i] for i in best_sents])
    return review


def peter_optimizer(bert, candidate_sents, config):
    config['opt_dict']['max_sentence_ind'] = len(candidate_sents)#software gore
    print("number of candidate sentences: ", len(candidate_sents))
    config['opt_dict']['eval_class'] = bert#straight-up software murder
    length_range = config['opt_dict']['length_range']
    len_X = config['opt_dict']['optimize_population']
    X = []
    sent_range = np.arange(0, len(candidate_sents), 1)
    for i in range(len_X):
        x_len = np.random.randint(length_range[0], length_range[1])
        '''x = []
        for j in range(x_len):
            x.append(np.random.randint(0, len(candidate_sents)))
        '''
        x = None
        if x_len <= len(candidate_sents):
            x = np.random.choice(sent_range, size = (x_len), replace = False).tolist()
        else:
            x = np.random.choice(sent_range, size = (x_len), replace = True).tolist()
        X.append(x)
    return config['opt_function'].optimize(X, config)

def evaluate(hyp_text, ref_text, hyp_enc, ref_enc):
    '''
    return evaluation of hypothesis text (our output) compared to
    reference text (gold standard)

    rouge score in form of tuple (f-score, precision, recall)
    cosine similarity in the form of float [0,1]

    param [hyp_text]: string of our summary text
    param [ref_text]: string of gold standard summary

    param [hyp_enc]: 1d numpy array of our summary embedding
    param [ref_enc]: 1d numpy array of gold standard embedding,
                        must be same dimensions as above
    '''
    eval_dict = ev.evaluate_rouge(hyp_text, ref_text)
    eval_dict['cos_sim'] = ev.evaluate_embeddings(hyp_enc, ref_enc)

    return eval_dict


def print_first_5(lst):
    '''
    print first 5 elements of lst

    param [lst]: very long list that would take too
    long to print fully
    '''
    print_str = '['
    for x in range(4):
        print_str += repr(lst[x])
        print_str += '; \n'

    print_str += '... ]'
    print(print_str)


def summarize_product(sentences, config):
    '''
    param [sentences]: the list of all tokenized review sentences in corpus
    '''
    encodings = encode(sentences, config)
    candidate_points = cluster(encodings, sentences, config)
    candidate_sents = decode(candidate_points, config)
    for c in candidate_sents:
        print(c)
    solution = optimize(candidate_sents, config)
    return solution

def summarize_dataset(config):
    load_config(config)
    results = {}
    for ix, sentences in config['dataset'].items():
        sents = list(filter(lambda x: len(x) > 40, sentences))
        output = summarize_product(sents, config)
        print('Final Output:', output)
        results[ix] = output
    with open(config['save_path'], 'w') as outfile:
        json.dump(results, outfile)

def summarize_datasets(config):
    for ix, dataset_path in enumerate(config['dataset_path_list']):
        config['dataset_path'] = dataset_path
        config['save_path'] = config['dataset_save_list'][ix]
        summarize_dataset(config)


if __name__ == "__main__":
    from optimization.finetune_bert_genetic_optimizer import GeneticBertOptimizer

    config = {
    'save_path' : 'data/electronics_results.json',
    'dataset_path' : 'autotransformer/data/electronics_dataset_1.pkl',
    'dataset_path_list': [],
    'dataset_save_list': [],
    'dataset' : None,

    'extractive' : False,
    'device' : None,

    'src_vocab_path' : 'models/electronics/src_vocab.pt',
    'src_vocab' : None,
    'trg_vocab_path' : 'models/electronics/trg_vocab.pt',
    'autoencoder_path': 'models/electronics/electronics_autoencoder_epoch7_weights.pt',
    'autoencoder' : None,
    'ae_batchsize': 5000,

    'density_parameter' : 2,
    'minimum_samples': 2,
    'min_clusters': 50,
    'max_acceptable_clusters': 200,
    'min_num_candidates': 250,

    'BERT_finetune_path' : 'models/electronics/finetune_electronics_mae_mk31.pt',
    'BERT_config_path' : 'models/electronics/finetune_electronics_mae_mk31config.json',
    'BERT_finetune_model' : None,
    'BERT_batchsize': 25,
    'length_penalty_order': 1.5,

    'opt_function' : GeneticBertOptimizer(),

    'opt_dict' : {
        'optimize_population': 96,#for optimization methods with a population at optimization estimates,
        #this is the number of optimization estimates used by the algorithm
        'n_elite': 5,
        'length_range': (5,20),
        'length_penalty_range': (0.4, 1.0),
        'p_replace': .33,
        'p_remove': .33,
        'p_add': .33,
        'prevent_dupe_sents': True,
        'max_iter': 10,
        'print_iters': 2
        }
    }
    summarize_dataset(config)
