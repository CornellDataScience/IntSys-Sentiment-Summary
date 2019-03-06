# autoencoder.py
# 2nd Mar. 2019
# IntSys-Summarization

import torch
import torch.nn as nn
import torch.nn.functional as F

t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device = t_device)), (torch.zeros(1, batch_size, self.hidden_size, device = t_device))

    def initialize_embeddings(self, weight_matrix, freeze = True):
        self.embedding = nn.Embedding.from_embedding(weight_matrix)
        self.embedding.weight.requires_grad = not freeze

class Decoder(nn.Module):

    def __init__(self, vocab_size, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, output_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers = 1)
        # input_size = ouput_size because we're using the prev. output as input
        self.lstm_out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 2) # dim = 2 because input is (1, batchsize, outputsize)

    def forward(self, input, hidden):
        # TODO; Consider batch size here?
        # TODO; Definitely allows one el. in the sequence, 
        #       because we don't know the length from the outset
        embedded = self.embedding(input)
        embedded_r = F.relu(embedded) # relu layer (because non-linear aspect on the prev. input)
        output, hidden = self.lstm(embedded_r, hidden)

        output = self.softmax(self.lstm_out(output))
        return output, hidden

    def initialize_embeddings(self, weight_matrix, freeze = True):
        self.embedding = nn.Embedding.from_embedding(weight_matrix)
        self.embedding.weight.requires_grad = not freeze

# Questions:
# 1. Do we use batch size in our specifications?
# 2. Share weights for encoder/decoder?
# 3. Is the decoder input both the cell state and final hidden state or
#    only the cell state
# 4. decoder - forward : how to accept sentences of varying lengths within
#    the same batch?


# Let' say for now --> we're not using pretrained embeddings

# TODO:
# 1. Figure out how to use pretrained embeddings (ignore for now)
# 2. Figure out how to pad sentences to get same batch length