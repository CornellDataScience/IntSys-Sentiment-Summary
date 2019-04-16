import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
                              
from tqdm import tqdm, trange
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from finetune_utils import InputExample, InputFeatures, convert_examples_to_features

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def process_data(data):
    examples = []
    for ix, review in enumerate(data):
        examples.append(InputExample(ix, review, label=.5))
    features = convert_examples_to_features(examples, [None], 512, tokenizer, 'regression')
    return features

def predict(model, data, device, batch_size=1):
    model.to(device)
    model.eval()
    predict_features = process_data(data)

    all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)

    predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(predict_data)
    dataloader = DataLoader(predict_data, sampler=sampler, batch_size=batch_size)

    predictions = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids = batch
        with torch.no_grad():
            preds = model(input_ids, segment_ids, input_mask, labels=None)

        if len(predictions) == 0:
            predictions.append(preds.detach().cpu().numpy())
        else:
            predictions[0] = np.append(
                predictions[0], preds.detach().cpu().numpy(), axis=0)

    return predictions[0]




    