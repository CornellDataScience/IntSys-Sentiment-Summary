import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from tqdm import tqdm, trange
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from .finetune_utils import InputExample, InputFeatures, convert_examples_to_features


class BERTpredictor():
    def __init__(self, config, sentences):
        self.model = config['BERT_finetune_model']
        self.batch_size = config['BERT_batchsize']
        self.device = config['device']
        self.sentences = [s.capitalize().replace(' .', '.') for s in sentences]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.length_penalty_range = config['opt_dict']['length_penalty_range']
        self.length_range = config['opt_dict']['length_range']
        self.penalty_order = config['length_penalty_order']


    def preprocess(self, candidate_ixs):
        cand_reviews = []
        for cand_ix, cand in enumerate(candidate_ixs):
            cand_rev = []
            for ix in cand:
                cand_rev.append(self.sentences[ix])
            full_review = ' '.join(cand_rev)
            rev_example = InputExample(cand_ix, full_review, label=0)
            cand_reviews.append(rev_example)

        features = convert_examples_to_features(cand_reviews, [None], 512, self.tokenizer, 'regression')
        return features

    def evaluate(self, candidate_ixs):
        predict_features = self.preprocess(candidate_ixs)

        all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)

        predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        sampler = SequentialSampler(predict_data)
        dataloader = DataLoader(predict_data, sampler=sampler, batch_size=self.batch_size)

        self.model.to(self.device)
        self.model.eval()

        predictions = []

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids = batch
            with torch.no_grad():
                preds = self.model(input_ids, segment_ids, input_mask, labels=None)

            if len(predictions) == 0:
                predictions.append(preds.detach().cpu().numpy())
            else:
                predictions[0] = np.append(
                    predictions[0], preds.detach().cpu().numpy(), axis=0)


        predictions = predictions[0].flatten()
        for i in range(len(candidate_ixs)):
            predictions[i] *= self.__get_length_penalty_factor(len(candidate_ixs[i]), self.length_penalty_order)

        return predictions

    def __get_length_penalty_factor(self, length, order):
        '''
        len_frac = (length - self.length_range[0])/(self.length_range[1] - self.length_range[0])
        return self.length_penalty_range[0] + (1-len_frac)*(self.length_penalty_range[1] - self.length_penalty_range[0])
        '''
        min_len = self.length_range[0]
        max_len = self.length_range[1]
        min_weight = self.length_penalty_range[0]
        max_weight = self.length_penalty_range[1]
        return min_weight + (1 - ((length - min_len)/(max_len - min_len))**order)*(max_weight - min_weight)
