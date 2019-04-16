from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from finetune_utils import InputExample, InputFeatures, convert_examples_to_features

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path):
    df = pd.read_csv(path)
    examples = []
    for ix, row in df.iterrows():
        examples.append(InputExample(ix, row.reviewText, label=row.helpful))
    return examples

def finetune(data_path, output_path, save_name, batch_size, n_epochs, learning_rate, warmup_proportion, gradient_accumulation_steps):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    batch_size = batch_size // gradient_accumulation_steps
    train_examples = load_data('data/sports_helpfulness.csv')
    num_train_optimization_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps) * n_epochs

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=learning_rate,
                     warmup=warmup_proportion,
                     t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    train_features = convert_examples_to_features(train_examples, [None], 512, tokenizer, 'regression', logger)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    model.train()
    loss_fct = MSELoss()
    for e in trange(int(n_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)
                
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % gradient_accumulation_steps == 0:
                print(loss)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
        print('Training Loss Epoch', str(e), ':', str(tr_loss/nb_tr_examples))
        
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_name = save_name + str(e) + '.pt'
        output_model_file = os.path.join(output_path, model_name)
        torch.save(model_to_save.state_dict(), output_model_file)
        config_name = save_name + str(e) +'config.json'
        output_config_file = os.path.join(output_path, config_name)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

"""MAKE SURE YOU CREATE A FOLDER CALLED MODELS BEFORE FINETUNING"""

output_path = 'models/'
save_name = 'finetune_sports'
data_path = 'data/sports_helpfulness.csv'

batch_size = 2
n_epochs = 2
learning_rate = 5e-5
warmup_proportion = .1
gradient_accumulation_steps = 1

finetune(data_path, output_path, save_name, batch_size, n_epochs, learning_rate, warmup_proportion, gradient_accumulation_steps)
