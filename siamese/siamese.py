# siamese.py
# 20th April 2019
# IntSys-Summarization

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext.data import BucketIterator

from tqdm import tqdm

import utils

# ========= CONSTANTS =========
t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
NUM_EPOCHS = 10
MARGIN = 0.01
LEARNING_RATE = 0.01
EMBEDDING_DIM = 300

# =============================

class SiameseAutoEncoder(nn.Module):

    def __init__(self, autoencoder, start_symbol):
        super(SiameseAutoEncoder, self).__init__()

        self.start_symbol = start_symbol
        self.autoencoder = autoencoder
        #self.anchor_autoencoder = autoencoder
        #self.left_autoencoder = autoencoder
        #self.right_autoencoder = autoencoder

    # anchor, pos,, neg. must be a list of indices
    # where each element shape (1, len)
    def forward(self, anchor, positive, negative):
        anchor_len, positive_len, negative_len = anchor.shape[1], positive.shape[1], negative.shape[1]
        anchor_embed, anchor_out = self.greedy_decode(self.autoencoder, anchor, 
                                                       Variable(torch.ones(1, 1, anchor_len)), anchor_len, 
                                                       self.start_symbol)
        pos_embed, pos_out = self.greedy_decode(self.autoencoder, positive, 
                                                Variable(torch.ones(1, 1, positive_len)), positive_len, 
                                                self.start_symbol)
        neg_embed, neg_out = self.greedy_decode(self.autoencoder, negative, 
                                                Variable(torch.ones(1, 1, negative_len)), negative_len, 
                                                self.start_symbol)

        return (anchor_embed, pos_embed, neg_embed), (anchor_out, pos_out, neg_out)

    # OpenNMT: Open-Source Toolkit for Neural Machine Translation
    # https://github.com/harvardnlp/annotated-transformer
    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        #print(src.shape)
        memory = model.encode(src, src_mask)[0:1, :, :]
        #print(memory.shape)f
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

# =================== TRAIN ===================
# anchor (n x d)
def train(model, criterion, anchor, positive, negative):
    batch_size = anchor.shape[1]
    (anchor_embed, pos_embed, neg_embed), _ = model(anchor, positive, negative)

    # calculate euclidean distance (can be altered if needed)
    dist_pos = torch.pow((anchor_embed - pos_embed), 2).sum(dim = 2).view(-1)
    dist_neg = torch.pow((anchor_embed - neg_embed), 2).sum(dim = 2).view(-1)

    #print(dist_pos.shape)
    #loss = criterion(torch.Tensor([dist_pos, dist_neg]).t, [-1] * batch_size)
    loss = criterion(dist_pos, dist_neg, torch.Tensor([-1] * batch_size))
    loss.backward()

    return loss.item()

def train_dataset(train_iterator, siamese_autoencoder, epochs, margin, learning_rate):
    siamese_optimizer = torch.optim.Adam(siamese_autoencoder.parameters(), lr=learning_rate)
    # just triplet loss for now
    criterion = nn.MarginRankingLoss(margin)

    all_losses = []
    for e in range(1, epochs + 1):
        epoch_loss = 0.0

        with tqdm(total=len(train_iterator), desc="Epoch % i" % e) as pbar:
            for i, batch in enumerate(train_iterator):
                siamese_optimizer.zero_grad()
                
                loss = train(siamese_autoencoder, criterion, 
                             batch.anc, batch.pos, batch.neg)

                siamese_optimizer.step()
                epoch_loss += loss

                pbar.update(len(batch))
                torch.cuda.empty_cache()

            all_losses.append(epoch_loss / len(train_iterator))

        torch.save(siamese_autoencoder.autoencoder, "siamese_ae_epoch_%d" % epoch)

    return all_losses

def main(model_path, train, val, test):
    train_iter, val_iter = BucketIterator.splits((train, val), batch_sizes=(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE), 
                                                 device=t_device, 
                                                 sort_key=lambda x: len(x.anc), sort_within_batch=False, 
                                                 repeat=False)

    autoencoder = load_autoencoder(model_path)
    all_losses = train_dataset(train_iter, autoencoder, NUM_EPOCHS, MARGIN, LEARNING_RATE)

    return all_losses, autoencoder

# =================== ANALYZE ===================
def run_embeddings(model, trg_vocab, sent_idxs):
    sentence_embeddings = np.zeros((len(sent_idxs), EMBEDDING_DIM))
    generated_sentences = [""] * len(sent_idxs) 

    for i, sent in enumerate(sent_idxs):
        sentence_embeddings[i, :], generated_sentences[i] = utils.generate_sentence(model, trg_vocab, sent)

    return sentence_embeddings, generated_sentences

# =================== HELPERS ===================
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def load_autoencoder(model_path):
    # eval() not used to ensure parameters can be updated
    return SiameseAutoEncoder(torch.load(model_path), 1)
