import argparse
import unicodedata
import re
import os
from os.path import join
import pickle
import yaml

class Voc:
"""
Args:
    size : vocabulary size
    lines : list of sentences
Attributes:
    index2word : a dict with key=index value=word
    word2index : a dict with key=word  value=index
"""
    def __init__(self, size=4, lines=[]):
        assert size >= 4
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD", 3:"UNK"}
        word2count = {}
        for l in lines:
            for word in l.split(' '):
                if word not in word2count:
                    word2count[word] = 1
                else:
                    word2count[word] += 1
        word2count = list(word2count.items())
        word2count.sort(key=lambda x: x[1], reverse=True)
        size = min(size, len(word2count))
        if len(lines):
            print("{} words trimmed to {} words".format(len(word2count), size))
        for i in range(size-4):
            self.index2word[i+4] = word2count[i][0]
        self.word2index = {v: k for k, v in self.index2word.items()}

    def getIndex(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["UNK"]

    def save2file(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.index2word, f, default_flow_style=False, allow_unicode=True)

    def load_file(self, path):
        with open(path, 'r') as f:
            self.index2word = yaml.load(f)
            self.word2index = {v: k for k, v in self.index2word.items()}
            self.size = len(self.index2word)

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# Let user select which dataset to preprocess
def selectSet(description, sets, directory):
    print('Datasets :')
    for idx, s in enumerate(sets):
        print(idx, ':', s)
    print(description)
    tr_set = input('Please enter the indices seperated by space:')
    tr_set = [join(directory, sets[int(t)]) for t in tr_set.split(' ')]
    return tr_set

# remove sequence with length greater than max_length
def filterPair(l, max_len):
    tmp = []
    for i, j in zip(l[0::2], l[1::2]):
        # input sequences need to preserve the last word for EOS_token, so '<' instead of '<='.
        if len(i.split(' ')) < max_len and \
                len(j.split(' ')) < max_len:
            tmp.append(i)
            tmp.append(j)
    print("{:.0f} pairs trimmed to {:.0f} pairs".format(len(l)/2, len(tmp)/2))
    return tmp

def readFile(files, max_len):
    lines = []
    for f in files:
        with open(f) as fs:
            lines += fs.read().splitlines()
    if max_len:
        lines = filterPair(lines, max_len)
    return lines

# Generate Vocabulary
def genVoc(config, sets):
    tr_set = selectSet('Select training dataset to generate vocabulary.', sets, config['data_dir'])
    tr_lines = readFile(tr_set, config['max_len'])
    if config['normalize']:
        tr_lines = [normalizeString(x) for x in tr_lines]
    voc = Voc(config['size'], tr_lines)
    voc.save2file(join(config['save_dir'], 'voc.pkl'))
    return voc

# Encode Corpus
def encode(config, sets, voc):
    def toIndex(line, eos):
        idx_seq = [voc.getIndex(x) for x in line.split(' ')]
        if eos:
            idx_seq += [1]
        return idx_seq
    enc_set = selectSet('Select dataset to encode.', sets, config['data_dir'])
    for f in enc_set:
        enc_lines = readFile([f], config['max_len'])
        if config['normalize']:
            enc_lines = [normalizeString(x) for x in enc_lines]
        enc_lines = [toIndex(x, idx%2) for (idx, x) in enumerate(enc_lines)]
        # sort the enc_lines by input(odd lines) length
        sorted_idx = sorted(range(0, len(enc_lines), 2), key=lambda k: len(enc_lines[k]), reverse=True)
        sorted_enc_lines = [(enc_lines[i], enc_lines[i+1]) for i in sorted_idx]
        # save to pickle file
        _, file_name = os.path.split(f)
        bn = os.path.splitext(file_name)[0]
        with open(join(config['save_dir'], bn+'.pkl'), 'wb') as fs:
            pickle.dump(sorted_enc_lines, fs)
        with open(join(config['save_dir'], 'config.yaml'), 'w') as fs:
            yaml.dump(config, fs, default_flow_style=False)

def preprocess(config):
    print("Preprocessing...")
    if not os.path.isdir(config['save_dir']):
        os.makedirs(config['save_dir'])
    sets = [d for d in os.listdir(config['data_dir'])]
    voc = genVoc(config, sets)
    encode(config, sets, voc)

# Config files with same preprocess parameters share preprocessed data
# Generate a look up table with 
#   key=(preprocess parameters) value=(preprocessed data directory)
def config_lookup(config, origin_save_dir):
    tmp = config['save_dir']
    config['save_dir'] = origin_save_dir
    lookup_table = {}
    lookup_table_path = join(origin_save_dir, 'config_data_map.pkl')
    if os.path.exists(lookup_table_path):
        with open(lookup_table_path, 'rb') as f:
            lookup_table = pickle.load(f)
    lookup_table[tuple(config.values())] = tmp
    with open(lookup_table_path, 'wb') as f:
        pickle.dump(lookup_table, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate vocabulary and turn corpus into sequence of indices.')
    parser.add_argument('--config', type=str, help='Config file')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'))
    _, config_name = os.path.split(args.config)
    config_name = os.path.splitext(config_name)[0]
    config = config['preprocess']
    origin_save_dir = config['save_dir'] # used in look-up table
    config['save_dir'] = join(config['save_dir'], config_name)
    preprocess(config)
    config_lookup(config, origin_save_dir)
