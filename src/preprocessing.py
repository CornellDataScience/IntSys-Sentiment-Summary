# preprocessing.py
# 19th Feb. 2019
# IntSys-Summarization

# ======== MODULES ========
from src.contractions import CONTRACTIONS
import dask.dataframe as dd
from dask.multiprocessing import get
import gensim
from multiprocessing import cpu_count
import pandas as pd
import re
import spacy
import unicodedata

spacy.prefer_gpu()

# ======== CONSTANTS ========
CONTRACTION_REGEX = re.compile('({})'.format('|'.join(CONTRACTIONS.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
CORE_COUNT = cpu_count()

nlp = spacy.load('en_core_web_sm') # TODO: Disable components if nec.

# ======== CLEANING TEXT ========

# Retain Numbers, Upper Case, Punctuation, Stop-Words
# No Stemming, Lemmatization
def clean_data(reviews):
    """ Returns (OOV, cleaned reviews) 
        where OOV : {<text> : [<row_idx>, ...], ...}"""
        
    ddask_reviewText = dd.from_pandas(reviews.reviewText, npartitions = CORE_COUNT)
    # substitute accents
    ddask_reviewText = dd.from_pandas(ddask_reviewText.map_partitions(lambda df : df.apply(lambda row : remove_accents(row))).compute(), 
                                      npartitions = CORE_COUNT)

    # expand contractions (replace with pycontractions?)
    ddask_reviewText = dd.from_pandas(ddask_reviewText.map_partitions(lambda df : df.apply(lambda row : expand_contractions(row, 
                                                                                                                            CONTRACTIONS, 
                                                                                                                            CONTRACTION_REGEX))).compute(), 
                                      npartitions = CORE_COUNT)

    # tokenize
    tokenized_text = ddask_reviewText.map_partitions(lambda df : df.apply(lambda row : nlp(row))).compute()

    # identify OOV words
    OOV_words = {} # {<token> : [<row_idx>, <row_idx>, ...], ..}
    for row_idx, doc in enumerate(tokenized_text):
        for i in range(len(doc)):
            if doc[i].text not in nlp.vocab:
                if doc[i].text in OOV_words:
                    OOV_words[doc[i].text].append(row_idx)
                else:
                    OOV_words[doc[i].text] = [row_idx]

    reviews = reviews.copy()
    reviews.drop(['reviewText', 'reviewerName', 'unixReviewTime'], inplace=True, axis = 1)
    reviews['reviewText'] = tokenized_text.apply(str)
    reviews['reviewDoc'] = tokenized_text

    # standardize date (DD-MM-YYYY)
    reviews['reviewTime'] = reviews.reviewTime.apply(lambda row : standardize_date(row))

    return OOV_words, reviews

def remove_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

# contraction_map : compiled regex
def expand_contractions(text, contraction_map, contraction_regex):

    def expand_contraction_match(matched_re):
        contraction = matched_re.group()
        expansion = contraction_map.get(contraction.lower())

        # should work for all but one case
        if contraction[0].isupper():
            return expansion[0].upper() + expansion[1:]
        else:
            return expansion

    return contraction_regex.sub(expand_contraction_match, text)

def standardize_date(exist_date):
    exist_date = exist_date.strip().split(" ")

    for i in range(len(exist_date) - 1):
        date_comp = exist_date[i].strip(", ") 
        date_comp = date_comp if len(date_comp) == 2 else "0" + date_comp
        exist_date[i] = date_comp

    return "{0} {1} {2}".format(exist_date[1], exist_date[0], exist_date[2].strip(", "))
