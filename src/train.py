# train.py
# 2nd Mar. 2019
# IntSys-Summarization

import numpy as np
import src.lang as lang
import torch
import torch.nn as nn
import torch.nn.functional as F

TEACHER_FORCING_RATIO = 0.7

t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# currently takes in only one input (shape: length, batch, ind_length)
# target : shape (length)
def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    input_batch_size = input.shape[1]
    encoder_hidden = encoder.init_hidden(batch_size = input_batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input, encoder_hidden)

    max_target_length = target.shape[0]

    decoder_input = torch.tensor([[lang.C_SOS_IDX] * input_batch_size], device=t_device)
    decoder_hidden = encoder_hidden #take final h, c of encoder as input
    outputs = torch.zeros(*input.shape, device=t_device)

    loss = 0

    for i in range(max_target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_top1 = decoder_output.topk(1, dim = 2)[1][0, :, 0]
        outputs[i] = decoder_top1

        use_teacher_forcing = np.random.uniform() < TEACHER_FORCING_RATIO

        decoder_input = target[i : i + 1] if use_teacher_forcing else decoder_top1.detach().view(1, -1)
        loss += criterion(decoder_output[0, :, :], target[i])
        

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), outputs

def train_dataset(train_iterator, encoder, decoder, epochs, learning_rate):
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    all_losses = []
    all_generations = []

    for e in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_generations = []

        for i, batch in enumerate(train_iterator):
            loss, ouputs = train(batch.src, batch.tgt,                  # not scalable
                                 encoder, decoder,
                                 encoder_optimizer, decoder_optimizer, 
                                 criterion)

            epoch_loss += loss
            epoch_generations.append(ouputs)

        all_losses.append(epoch_loss)
        all_generations.append(epoch_generations)

    return all_losses, all_generations