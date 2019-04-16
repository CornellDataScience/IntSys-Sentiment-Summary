# autoencoder.py
# 20th Mar. 2019
# IntSys-Summarization

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SiameseAutoEncoder(nn.Module):

    def __init__(self, autoencoder, start_symbol):
        super(SiameseAutoEncoder, self).__init__()

        self.start_symbol = start_symbol
        self.anchor_autoencoder = autoencoder
        self.left_autoencoder = autoencoder
        self.right_autoencoder = autoencoder

    # anchor, pos,, neg. must be a list of indices
    # where each element shape (1, len)
    def forward(self, anchor, positive, negative):
        anchor_len, positive_len, negative_len = anchor.shape[1], positive.shape[1], negative.shape[1]
        anchor_embed, anchour_out = self.greedy_decode(self.anchor_autoencoder, anchor, 
                                                       Variable(torch.ones(1, 1, anchor_len)), anchor_len, 
                                                       self.start_symbol)
        pos_embed, pos_out = self.greedy_decode(self.left_autoencoder, positive, 
                                                Variable(torch.ones(1, 1, positive_len)), positive_len, 
                                                self.start_symbol)
        neg_embed, neg_out = self.greedy_decode(self.right_autoencoder, negative, 
                                                Variable(torch.ones(1, 1, negative_len)), negative_len, 
                                                self.start_symbol)

        return (anchor_embed, pos_embed, neg_embed), (anchor_out, pos_out, neg_out)

    # OpenNMT: Open-Source Toolkit for Neural Machine Translation
    # https://github.com/harvardnlp/annotated-transformer
    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        memory = model.encode(src, src_mask)
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

    return criterion(torch.Tensor([dist_pos, dist_neg]).t, [-1] * batch_size)

# Define Loss
# loss = nn.MarginRankingLoss(margin)

# =================== HELPERS ===================
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



