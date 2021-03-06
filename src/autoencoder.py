# autoencoder.py
# 2nd Mar. 2019
# IntSys-Summarization

import torch
import torch.nn as nn
import torch.nn.functional as F

import lang

t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def initialize_shared_embeddings(weight_matrix, freeze = True):
    shared_embedding = nn.Embedding.from_pretrained(weight_matrix)
    shared_embedding.weight.requires_grad = not freeze
    shared_embedding.to(device = t_device)

    return shared_embedding

class Encoder(nn.Module):

    def __init__(self, vocab_size, input_size, hidden_size, embedding_layer):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = embedding_layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 1)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size, device = t_device)), 
               (torch.zeros(1, batch_size, self.hidden_size, device = t_device))

class Decoder(nn.Module):

    def __init__(self, vocab_size, output_size, hidden_size, embedding_layer):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = embedding_layer
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

def AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, use_teacher_forcing):
        input_batch_size = input.shape[1]
        encoder_hidden = self.encoder.init_hidden(batch_size = input_batch_size)

        encoder_output, encoder_hidden = self.encoder(input, encoder_hidden)

        max_target_length = target.shape[0]

        decoder_input = torch.tensor([[lang.C_SOS_IDX] * input_batch_size], device=t_device)
        decoder_hidden = encoder_hidden #take final h, c of encoder as input
        outputs = torch.zeros(*input.shape, device=t_device)

        for i in range(max_target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_top1 = decoder_output.topk(1, dim = 2)[1][0, :, 0]
            outputs[i] = decoder_top1

            decoder_input = target[i : i + 1] if use_teacher_forcing else decoder_top1.detach().view(1, -1)

        return encoder_hidden, outputs



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