# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch
import torch.nn as nn

from .functional import clones
from .layer_norm import LayerNorm


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)
        #self.head = head

    def forward(self, x, x_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, x_mask)
        x = self.norm(x)
        n_sents, n_tokens, n_dims = x.shape
        sent_vec = x[:,0,:]
        return sent_vec.repeat(n_tokens,1,1).permute(1, 0, 2)

    # def forward(self, x, x_mask):
    #     """
    #     Pass the input (and mask) through each layer in turn.
    #     """
    #     for layer in self.layers:
    #         x = layer(x, x_mask)
    #     x = self.norm(x)
    #     n_sents, n_tokens, n_dims = x.shape
    #     sent_means = torch.mean(x, 1)
    #     return sent_means.repeat(n_tokens,1,1).permute(1, 0, 2)
