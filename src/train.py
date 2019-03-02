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

# currently takes in only one input (shape: length, batch, ind_length)
# target : shape (length)
def train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.init_hidden(batch_size = 1)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_output, encoder_hidden = encoder(input.view(-1, 1), encoder_hidden)

    decoder_input = torch.tensor([[lang.C_SOS_IDX]], device=t_device)
    decoder_hidden = encoder_hidden #take final h, c of encoder as input
    target_length = target.shape[0]

    use_teacher_forcing = np.random.uniform() < TEACHER_FORCING_RATIO

    loss = 0

    generated_sent = []
    if use_teacher_forcing:
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output[0, :, :], target[i])
            decoder_input = target[i].view(1, 1)

            generated_sent.append(decoder_output.topk(1)[1].item())
    else:
        for i in range(target_length): # can only go uptil target length
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topi = decoder_output.topk(1)[1]

            loss += criterion(decoder_output[0, :, :], target[i])
            decoder_input = topi.squeeze().detach().view(1, 1) # detach from graph, no gradient
            
            generated_sent.append(topi.item())

            if decoder_input.item() == lang.C_EOS:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return ((loss.item() / target_length), generated_sent)

def train_dataset(train_set, encoder, decoder, learning_rate, epochs):
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()
    all_losses = []
    all_generations = []

    for epoch in range(1, epochs + 1):
        epoch_loss = []
        epoch_generations = []
        for sample_idx in range(len(train_set)):
            sample = torch.tensor(train_set[sample_idx]).view(-1, 1)
            loss, generated_sent = train(sample, sample, encoder, decoder, 
                         encoder_optimizer, decoder_optimizer, criterion)

            epoch_loss.append(loss)
            epoch_generations.append(generated_sent)

        all_losses.append(epoch_loss)
        all_generations.append(epoch_generations)

    return all_losses, all_generations