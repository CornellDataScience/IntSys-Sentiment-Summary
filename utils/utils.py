# utils.py
# 24th April 2019
# IntSys-Summarization

import numpy as np
from nltk.corpus import stopwords
import spacy
import torch

from torch.autograd import Variable

# =============== CONSTANTS ===============
STOP_WORDS = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_lg")

# =============== SENT. PROC. ===============
def create_sentence_set(sentences):
    sentence_sets = []
    for sent in sentences:
        sent_set = set()
        for word in sent.strip().split():
            if word not in STOP_WORDS:
                sent_set.add(word)

        sentence_sets.append(sent_set)

    return sentence_sets

def create_spacy_docs(sentences):
    return [nlp(sent.strip()) for sent in sentences]

def create_spacy_text(docs):
    sentences = []
    for d in docs:
        sentences.append(" ".join([token.text.lower() for token in d]))

    return sentences

def indexify_sentences(src_vocab, sentences):
    sent_idxs = []
    for sent in sentences:
        idxs = torch.LongTensor([[src_vocab.stoi['<cls>']]  +
                                 [src_vocab.stoi[word] for word in sent.strip().split()]])
        sent_idxs.append(idxs)

    return sent_idxs

# =============== MODEL ===============
# OpenNMT: Open-Source Toolkit for Neural Machine Translation
# https://github.com/harvardnlp/annotated-transformer
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    #print(src.shape)
    memory = model.encode(src, src_mask)[0:1, :, :]
    #print(memory.shape)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return memory, ys

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# src : list of indices
def generate_sentence(model, trg_vocab, src):
    embedding, decode_idx = greedy_decode(model, src, Variable(torch.ones(1, 1, src.shape[1])),
                                          src.shape[1], start_symbol=trg_vocab.stoi["<s>"])
    
    return embedding.detach().numpy()[0, 0:1, :], " ".join([trg_vocab.itos[idx] for idx in decode_idx[0]])
