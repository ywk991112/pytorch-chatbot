import argparse
from train import trainEpochs
from evaluate import runTest
from load import Voc


def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', action='store_true', help='Train the model')
    parser.add_argument('-te', '--test', help='Test the model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-e', '--epoch', type=int, default=100000, help='Train the model with e epochs')
    parser.add_argument('-p', '--print', type=int, default=5000, help='Print every p epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-s', '--save', type=float, default=10000, help='Save every s epochs')

    args = parser.parse_args()
    return args

def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    if test:
        layers, hidden = filename[-2].split('_')
        n_layers = int(layers.split('-')[0])
        hidden_size = int(hidden)
        return n_layers, hidden_size, reverse
    return reverse

def run(args):
    reverse, fil, n_epochs, print_every, save_every, learning_rate, n_layers, hidden_size, batch_size, beam_size, input = \
        args.reverse, args.filter, args.epoch, args.print, args.save, args.learning_rate, \
        args.layer, args.hidden, args.batch_size, args.beam, args.input
    if args.train:
        trainEpochs(reverse, n_epochs, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every)
    elif args.load:
        reverse = parseFilename(args.load)
        trainEpochs(reverse, n_epochs, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every, loadFilename=args.load)
    elif args.test:
        n_layers, hidden_size, reverse = parseFilename(args.test, True)
        runTest(n_layers, hidden_size, reverse, args.test, beam_size, input)


if __name__ == '__main__':
    args = parse()
    run(args)
