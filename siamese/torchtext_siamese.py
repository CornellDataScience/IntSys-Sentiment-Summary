# torch_text_process.py
# 2nd Mar. 2019
# IntSys-Summarization

import dill
import os

import torch
import torch.nn as nn
from torchtext import data, datasets
from torchtext.vocab import Vectors

# CONSTANTS
SOS = 'SOS'
EOS = 'EOS'

VECTOR_MODEL = Vectors(name='glove.6B.300d.txt', cache=os.path.join(os.getcwd(), "..", "autotransformer", "data"))

FIELDS_ROOT_PATH = os.path.join("data")
ANC_FIELD_PATHS = ("anc", os.path.join(FIELDS_ROOT_PATH, "anc"))
POS_FIELD_PATHS = ("pos", os.path.join(FIELDS_ROOT_PATH, "pos"))
NEG_FIELD_PATHS = ("neg", os.path.join(FIELDS_ROOT_PATH, "neg"))

def whitespace_tokenizer(text):
    text = text.strip().lower().split(" ")
    if text[-1] == ".":
        return text[:-1]
    else:
        return text

# TODO : Pass Vector Model
def load_torchtext_datasets(data_path, rel_train, rel_val, rel_test):
    if not os.path.exists(ANC_FIELD_PATHS[1]):
        ANC = data.Field(
                sequential = True,
                tokenize = whitespace_tokenizer,
                init_token = SOS,
                eos_token = EOS,
            )
    else:
        with open(ANC_FIELD_PATHS[1], 'rb') as dill_file:
            ANC = dill.load(dill_file)
        # Do we have to call build_vocab again? Because that would defeat the purp.

    if not os.path.exists(POS_FIELD_PATHS[1]):
        POS = data.Field(
                sequential = True,
                tokenize = whitespace_tokenizer,
                init_token = SOS,
                eos_token = EOS,
            )
    else:
        with open(POS_FIELD_PATHS[1], 'rb') as dill_file:
            POS = dill.load(dill_file)
        # Do we have to call build_vocab again? Because that would defeat the purp.

    if not os.path.exists(NEG_FIELD_PATHS[1]):
        NEG = data.Field(
                sequential = True,
                tokenize = whitespace_tokenizer,
                init_token = SOS,
                eos_token = EOS,
            )
    else:
        with open(NEG_FIELD_PATHS[1], 'rb') as dill_file:
            NEG = dill.load(dill_file)
        # Do we have to call build_vocab again? Because that would defeat the purp.

    train, val, test = data.TabularDataset.splits(
        path = data_path, train = rel_train, validation = rel_val, test = rel_test, 
        format='csv', fields=[(ANC_FIELD_PATHS[0], ANC), 
                              (POS_FIELD_PATHS[0], POS), 
                              (NEG_FIELD_PATHS[0], NEG)], skip_header = True)

    # Build ANC and POS Vocab if not preloaded & save fields
    if not os.path.exists(ANC_FIELD_PATHS[1]):
        ANC.build_vocab(train, vectors = VECTOR_MODEL) #TODO: Add vectors? How do we ensure that made-up-tokens are there?
        #ANC.build_vocab(train)

        with open(ANC_FIELD_PATHS[1], 'wb+') as dill_file:
            dill.dump(ANC, dill_file)

    if not os.path.exists(POS_FIELD_PATHS[1]):
        POS.build_vocab(train, vectors = VECTOR_MODEL) #TODO: Add vectors? How do we ensure that made-up-tokens are there?
        #POS.build_vocab(train)

        with open(POS_FIELD_PATHS[1], 'wb+') as dill_file:
            dill.dump(POS, dill_file)

    if not os.path.exists(NEG_FIELD_PATHS[1]):
        NEG.build_vocab(train, vectors = VECTOR_MODEL) #TODO: Add vectors? How do we ensure that made-up-tokens are there?
        #POS.build_vocab(train)

        with open(NEG_FIELD_PATHS[1], 'wb+') as dill_file:
            dill.dump(NEG, dill_file)

    return (train, val, test), (ANC, POS, NEG)

# remodelled to accomodate existing vocab
def load_dataset_with_vocab(vocab_path, data_path, rel_train, rel_val, rel_test):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    CLS_WORD = '<cls>'
    SRC = data.Field(tokenize=whitespace_tokenizer, lower=True, init_token=CLS_WORD, pad_token=BLANK_WORD)
    #TGT = data.Field(tokenize=whitespace_tokenizer, lower=True, init_token=BOS_WORD,
    #                eos_token=EOS_WORD, pad_token=BLANK_WORD)

    vocab = torch.load(vocab_path)
    SRC.vocab = vocab

    train, val, test = data.TabularDataset.splits(
        path = data_path, train = rel_train, validation = rel_val, test = rel_test, 
        format='csv', fields=[(ANC_FIELD_PATHS[0], SRC), 
                              (POS_FIELD_PATHS[0], SRC), 
                              (NEG_FIELD_PATHS[0], SRC)], skip_header = True)

    return (train, val, test), SRC