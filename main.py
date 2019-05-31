import argparse
import yaml
import os
import pickle
from src.solver import Solver
from os.path import join

def parse():
    parser = argparse.ArgumentParser(description='Attention Seq2Seq Chatbot')
    parser.add_argument('-c', '--config', type=str, help='Path to the config file.')
    parser.add_argument('-te', '--test', action='store_true', help='Test the saved model')
    parser.add_argument('-l', '--load', type=str, help='Continue training with saved checkpoint.')
    parser.add_argument('-i', '--input', action='store_true', help='Test the model by input the sentence')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU for training')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs for parallel training')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    print(config)
    _, config_name = os.path.split(args.config)
    config_name = os.path.splitext(config_name)[0]
    config['solver']['log_dir']  = join(config['solver']['log_dir'], config_name)
    config['solver']['save_dir'] = join(config['solver']['save_dir'], config_name)
    with open(join(config['preprocess']['save_dir'], 'config_data_map.pkl'), 'rb') as f:
        lookup_table = pickle.load(f)
    config['preprocess']['save_dir'] = lookup_table[tuple(config['preprocess'].values())]

    solver = Solver(args, config)
    if args.test:
        solver.test()
    else:
        solver.train()
