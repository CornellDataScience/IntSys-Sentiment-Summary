'''The AutoTransformer Model

Intelligent systems - Sentiment Summarization Spring 2019

This is an implemementation of the autotransformer model, a model that
creates a fixed length vector representation of an input piece of text
through an encoder and then tries to reconstruct it using a decoder.

Implementation adapated from https://github.com/SamLynnEvans/Transformer'''

import math
import copy
import torch
import numpy as np
import torch.nn as nn

################### Embedding ###################
class Embedder(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)

    def forward(self, X):
        return self.embed(X)

class PositionalEncoder(nn.Module):
    def __init__(self, model_dim, max_seq_len=50):
        super().__init__()
        
        #Create positional embedding matrix
        pe = torch.zeros(max_seq_len, model_dim)
        for tok_pos in range(max_seq_len):
            for dim in range(0, model_dim, 2): #For dimension of token embedding
                pe[tok_pos, dim] = math.sin(tok_pos / (10000 ** ((2 * dim) / model_dim)))
                pe[tok_pos, dim+1] = math.cos(tok_pos / (10000 ** ((2 * (dim+1)) / model_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) #So the optimizer doesn't update during training

    def forward(self, X):
        X = X * math.sqrt(self.model_dim) #Make the token embedding signal stronger

        seq_len = X.size(1)
        X = X + torch.Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return X


################# Attention ###################

def attention(q, k, v, k_dim, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(k_dim)

    if mask:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0, -1e9)

    scores = nn.functional.softmax(scores, dim=-1)

    if dropout:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, model_dim, drop_rate=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.k_dim = model_dim // heads
        self.h = heads

        self.q_linear = nn.Linear(model_dim, model_dim)
        self.v_linear = nn.Linear(model_dim, model_dim)
        self.k_linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.output_layer = nn.Linear(model_dim, model_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        #Apply linear layers and split into h heads
        k = self.k_linear(k).view(batch_size, -1, self.h, self.k_dim) #TODO: apply dropout here?
        q = self.q_linear(q).view(batch_size, -1, self.h, self.k_dim)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.k_dim)

        #Get tensors with shape batch_size x h x seq_len x model_dim
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        #Calculate attention scores, concatenate, and apply final linear layer
        scores = attention(q, k, v, self.k_dim, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.model_dim)

        output = self.output_layer(concat)
        return output


####################### Feed Forward and Normalize #####################

class FeedForward(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=1024, drop_rate=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(model_dim, feed_forward_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.linear_2 = nn.Linear(feed_forward_dim, model_dim)

    def forward(self, X):
        X = self.dropout(nn.functional.relu(self.linear(X)))
        X = self.linear_2(X)
        return X

class BatchNorm(nn.Module):
    def __init__(self, model_dim, eps=-1e6):
        super().__init__()
    
        self.size = model_dim
        #Parameters for learning normalization
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, X):
        scaled_norm = self.alpha * (X - X.mean(dim=-1, keepdim=True)) 
        norm = scaled_norm / (X.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

######################## Encoder/Decoder #################################

class EncoderLayer(nn.module):
    def __init__(self, model_dim, heads, drop_rate=0.1):
        super().__init__()
        self.norm_1 = BatchNorm(model_dim)
        self.norm_2 = BatchNorm(model_dim)

        self.attn = MultiHeadAttention(heads, model_dim)
        self.ff = FeedForward(model_dim)

        self.dropout_1 = nn.Dropout(drop_rate)
        self.dropout_2 = nn.Dropout(drop_rate)

    def forward(self, X, mask):
        X2 = self.norm_1(X)
        X = X + self.dropout_1(self.attn(X2, X2, X2, mask))

        X2 = self.norm_2(X)
        X = X + self.dropout_2(self.ff(X2))
        return X

#TODO: detemine how to combine output encodings to single vector
class EncoderHead(nn.module):
    def __init__(self, model_dim):
        self.norm = BatchNorm(model_dim)
        self.weights = nn.Parameter(torch.ones())

class DecoderLayer(nn.module):
    def __init__(self, model_dim, heads, drop_rate=0.1):
        super().__init__()
        self.norm_1 = BatchNorm(model_dim)
        self.norm_2 = BatchNorm(model_dim)
        self.norm_3 = BatchNorm(model_dim)

        self.dropout_1 = nn.Dropout(drop_rate)
        self.dropout_2 = nn.Dropout(drop_rate)
        self.dropout_3 = nn.Dropout(drop_rate)

        self.attn_1 = MultiHeadAttention(heads, model_dim)
        self.attn_2 = MultiHeadAttention(heads, model_dim)

        self.ff = FeedForward(model_dim).cuda()

    def forward(self, X, enc_outputs, scr_mask, trg_mask):
        X2 = self.norm_1(X)
        X = X + self.dropout_1(self.attn_1(X2, X2, X2, trg_mask))
        X2 = self.norm_2(X)
        X = X + self.dropout_2(self.attn_2(X2, enc_outputs, enc_outputs, scr_mask))
        X2 = self.norm_3(X)
        X = X + self.dropout_3(self.ff(X2))
        return X

def get_clones(module, N_clones):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N_clones)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, model_dim, N_layers, heads):
        super().__init__()
        self.N = N_layers
        self.embed = Embedder(vocab_size, model_dim)
        self.pe = PositionalEncoder(model_dim)
        self.layers = get_clones(EncoderLayer(model_dim, heads), N_layers)
        self.norm = BatchNorm(model_dim)

    def forward(self, src, mask):
        X = self.embed(src)
        X = self.pe(X)
        for i in range(self.N):
            X = self.layers[i](X, mask)
        return self.norm(X)

class Decoder(nn.Module):
    def __init__(self, vocab_size, model_dim, N_layers, heads):
        super().__init__()
        self.N = N_layers
        self.embed = Embedder(vocab_size, model_dim)
        self.pe = PositionalEncoder(model_dim)
        self.layers = get_clones(DecoderLayer(model_dim, heads), N_layers)
        self.norm = BatchNorm(model_dim)

    def forward(self, trg, enc_outputs, src_mask, trg_mask):
        X = self.embed(trg)
        X = self.pe(X)
        for i in range(self.N):
            X = self.layers[i](X, enc_outputs, src_mask, trg_mask)
        return self.norm(X)

###################### AutoTransformer #############################

class AutoTransformer(nn.module):
    def __init__(self, src_vocab, trg_vocab, model_dim, N_elayers, N_dlayers, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, model_dim, N_elayers, heads)
        self.decoder = Decoder(trg_vocab, model_dim, N_dlayers, heads)
        self.output_layer = nn.Linear(model_dim, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_outputs = self.encoder(src, src_mask)
        dec_outputs = self.decoder(trg, enc_outputs, src_mask, trg_mask)
        output = self.output_layer(dec_outputs)
        return output