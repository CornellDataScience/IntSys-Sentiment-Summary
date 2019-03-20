#!/usr/bin/env python
# -*- coding: utf-8 -*-
# date: 2018-12-02 20:54
import spacy
import torch
import torch.nn as nn
from torchtext import data, datasets

from transformer.flow import make_model, batch_size_fn, run_epoch
from transformer.greedy import greedy_decode
from transformer.label_smoothing import LabelSmoothing
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.my_iterator import MyIterator, rebatch
from transformer.noam_opt import NoamOpt

# GPUs to use
devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]  # Or use [0, 1] etc for multiple GPUs
spacy_en = spacy.load('en')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def get_dataset(train_path, val_path, MIN_FREQ):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                    eos_token=EOS_WORD, pad_token=BLANK_WORD)


    data_fields = [('src', SRC), ('trg', TGT)]

    train = data.TabularDataset(format='csv', path= train_path, fields=data_fields)
    val = data.TabularDataset(format='csv', path= val_path, fields=data_fields)

    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)
    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    return train, val, TGT, SRC, pad_idx, EOS_WORD, BOS_WORD, BLANK_WORD

def train(
    train_path,
    val_path, 
    save_path,
    n_layers = 6,
    model_dim = 512,
    feedforward_dim = 2048,
    n_heads = 8,
    dropout_rate = 0.1,
    n_epochs = 10,
    max_len = 60,
    min_freq = 10,
    max_val_outputs = 20):

    train, val, TGT, SRC, pad_idx, EOS_WORD, BOS_WORD, BLANK_WORD = get_dataset(train_path, val_path, min_freq)

    model = make_model(len(SRC.vocab), len(TGT.vocab),
                         n=n_layers, d_model=model_dim,
                         d_ff=feedforward_dim, h=n_heads,
                         dropout=dropout_rate)
    model.cuda()
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 600  # Was 12000, but I only have 12 GB RAM on my single GPU.
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0, repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(n_epochs):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
        save_name = save_path + '_epoch' + str(epoch) + '.pt'
        torch.save(model.state_dict(), save_name)
        model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                         MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))
        print(loss)

    for i, batch in enumerate(valid_iter):
        if i > max_val_outputs:
            break
        src = batch.src.transpose(0, 1)[:1].cuda()
        src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2).cuda()
        out = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=TGT.vocab.stoi[BOS_WORD])
        print('Translation:', end='\t')
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == EOS_WORD:
                break
            print(sym, end=' ')
        print()
        print('Target:', end='\t')
        for j in range(batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[j, 0]]
            if sym == EOS_WORD:
                break
            print(sym, end=' ')
        print()


if __name__ == '__main__':
    train_path = 'data/test_train.csv'
    val_path = 'data/test_val.csv'
    save_path = 'models/copy_transfomer'
    n_layers = 6 #for encoder and decoder
    model_dim = 512
    feedforward_dim = 2048
    n_heads = 8
    dropout_rate = 0.1
    n_epochs = 3
    max_len = 60
    min_freq = 3
    max_val_outputs = 20
    train(train_path, val_path, save_path, n_layers, model_dim, feedforward_dim,
    n_heads, dropout_rate, n_epochs, max_len, min_freq, max_val_outputs)
