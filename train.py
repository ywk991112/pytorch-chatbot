import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn

import itertools
import random
import math
import os
from tqdm import tqdm
from load import loadPrepareData
from load import SOS_token, EOS_token, PAD_token
from model import EncoderRNN, LuongAttnDecoderRNN
from config import MAX_LENGTH, teacher_forcing_ratio, save_dir

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

cudnn.benchmark = True
#############################################
# Training
#############################################

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1) # [64, 1]

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            loss += F.cross_entropy(decoder_output, target_variable[t], ignore_index=EOS_token)

    loss.backward()

    clip = 50.0
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_target_len 


def trainIters(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):

    voc, pairs = loadPrepareData(corpus)

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(save_dir, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
        torch.save(training_batches, os.path.join(save_dir, 'training_data', corpus_name,
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                        filename(reverse, 'training_batches'), \
                                                                        batch_size)))
    # model
    checkpoint = None
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers, dropout)
    attn_model = 'dot'
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers, dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']

    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))
