import argparse
from train import trainIters
from evaluate import runTest

def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-tr', '--train', help='Train the model with corpus')
    parser.add_argument('-te', '--test', help='Test the saved model')
    parser.add_argument('-l', '--load', help='Load the model and train')
    parser.add_argument('-c', '--corpus', help='Test the saved model with vocabulary of the corpus')
    parser.add_argument('-r', '--reverse', action='store_true', help='Reverse the input sequence')
    parser.add_argument('-f', '--filter', action='store_true', help='Filter to small training data set')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('-it', '--iteration', type=int, default=10000, help='Train the model with it iterations')
    parser.add_argument('-p', '--print', type=int, default=100, help='Print every p iterations')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('-la', '--layer', type=int, default=1, help='Number of layers in encoder and decoder')
    parser.add_argument('-hi', '--hidden', type=int, default=256, help='Hidden size in encoder and decoder')
    parser.add_argument('-be', '--beam', type=int, default=1, help='Hidden size in encoder and decoder')
    parser.add_argument('-s', '--save', type=int, default=500, help='Save every s iterations')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.1, help='Dropout probability for rnn and dropout layers')

    args = parser.parse_args()
    return args

def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse

def run(args):
    reverse, fil, n_iteration, print_every, save_every, learning_rate, \
        n_layers, hidden_size, batch_size, beam_size, inp, dropout = \
        args.reverse, args.filter, args.iteration, args.print, args.save, args.learning_rate, \
        args.layer, args.hidden, args.batch_size, args.beam, args.input, args.dropout
    if args.train and not args.load:
        trainIters(args.train, reverse, n_iteration, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every, dropout)
    elif args.load:
        n_layers, hidden_size, reverse = parseFilename(args.load)
        trainIters(args.train, reverse, n_iteration, learning_rate, batch_size,
                    n_layers, hidden_size, print_every, save_every, dropout, loadFilename=args.load)
    elif args.test:
        n_layers, hidden_size, reverse = parseFilename(args.test, True)
        runTest(n_layers, hidden_size, reverse, args.test, beam_size, inp, args.corpus)


if __name__ == '__main__':
    args = parse()
    run(args)
