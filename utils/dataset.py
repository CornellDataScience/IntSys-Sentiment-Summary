# dataset.py
# 27th April 2019
# IntSys-Summarization

import os
import pickle

import utils

class Dataset(object):

    # FIELDS
    #products : [prod1, prod2, ...]

    # ARGUMENTS
    # prod_to_idxs : {<prod_id> : [1, 3, 5, ...], ...}
    # idxs_to_sents : {1 : [<sent1>, <sent2>], ...}
    # prod_most_helpful : {<prod_id> : <review>}

    # TODO : Add Most Helpful Review

    def __init__(self, prod_to_idxs=None, idxs_to_sents=None, prod_most_helpful=None, prods_path=None):
        self.products = []

        if prods_path is not None:
            for f in os.listdir(prods_path):
                file_full_path = os.path.join(prods_path, f)
                if os.path.isfile(file_full_path):
                    self.products.append(pickle.load(open(file_full_path, 'rb')))
        else:
            print("Not using any directory, generating product objects.")

            #create prod_to_sents
            prod_to_sents = {p_name : [idxs_to_sents[i] for i in prod_to_idxs[p_name]] for p_name in prod_to_idxs.keys()}
            for p_name, lists in prod_to_sents.items():
                #print(lists[:10])
                f_list = []
                for sublist in lists:
                    #print(sublist)
                    #for s in sublist:
                    f_list.append(sublist)

                #print(f_list[:10])

                prod_to_sents[p_name] = f_list

            #print(prod_to_sents["B007WTAJTO"])
            for prod_idx, sent_lst in prod_to_sents.items():
                self.products.append(Product(prod_idx, sent_lst, None))

    def save_products(self, path):
        for product in self.products:
            pickle.dump(product, open(os.path.join(path, "%s.pkl" %product.product_id), 'wb'))
        print("Saved all Products")

class Product(object):
    def __init__(self, product_id, sentences, most_helpful_review):
        #print(product_id)
        self.product_id = product_id
        self.sentences = self.preprocess(sentences)
        self.most_helpful_review = most_helpful_review
        #print(self.sentences)

    def preprocess(self, sentences):
        #print(sentences[0])
        sent_docs = utils.create_spacy_docs(sentences)
        return utils.create_spacy_text(sent_docs)