# autoencoder.py
# 20th Mar. 2019
# IntSys-Summarization

import torch
import torch.nn as nn
import torch.nn.functional as F

t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# autoencoder(example) returns the encoder_embedding, decoder_output

class SiameseAutoEncoder(nn.Module):

    def __init__(self, autoencoder):
        super(SiameseAutoEncoder, self).__init__()

        self.anchor_autoencoder = autoencoder
        self.left_autoencoder = autoencoder
        self.right_autoencoder = autoencoder

    def forward(self, anchor, positive, negative):
        anchor_embed, anchour_out = self.anchor_autoencoder(anchor)
        pos_embed, pos_out = self.left_autoencoder(positive)
        neg_embed, neg_out = self.right_autoencoder(negative)

        return (anchor_embed, pos_embed, neg_embed), (anchor_out, pos_out, neg_out)

# anchor (n x d)
def train(model, criterion, anchor, positive, negative):
    (anchor_embed, pos_embed, neg_embed), _ = model(anchor, positive, negative)

    # calculate euclidean distance (can be altered if needed)
    dist_pos = torch.pow((anchor_embed - pos_embed), 2).sum(dim = 1)
    dist_neg = torch.pow((anchor_embed - neg_embed), 2).sum(dim = 1)

    # TODO: restrictions on dimensions to be passed onto criterion?
    return criterion() # TODO

# Define Loss
# loss = nn.MarginRankingLoss(margin)



