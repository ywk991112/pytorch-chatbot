import argparse
import unicodedata
import re
import os
from os.path import join
import pickle
import yaml

class Voc:
    def __init__(self, size, lines):
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
        self.size = min(size, len(word2count))
        for i in range(self.size-4):
            self.index2word[i+4] = word2count[i][0]
        self.word2index = {v: k for k, v in self.index2word.items()}

    def getIndex(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["UNK"]

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

def readFile(files):
    lines = []
    for f in files:
        with open(f) as fs:
            lines += fs.read().splitlines()
    return lines

# Generate Vocabulary
def genVoc(args, sets):
    tr_set = selectSet('Select training dataset to generate vocabulary.', sets, args.data_dir)
    tr_lines = readFile(tr_set)
    if args.normalize:
        tr_lines = [normalizeString(x) for x in tr_lines]
    voc = Voc(args.size, tr_lines)
    with open(join(args.save_dir, 'voc.pkl'), 'wb') as f:
        pickle.dump(voc, f)
    return voc

# Encode Corpus
def encode(args, sets, voc):
    def toIndex(line):
        return [voc.getIndex(x) for x in line.split(' ')]
    enc_set = selectSet('Select dataset to encode.', sets, args.data_dir)
    for f in enc_set:
        enc_lines = readFile([f])
        if args.normalize:
            enc_lines = [normalizeString(x) for x in enc_lines]
        enc_lines = [toIndex(x) for x in enc_lines]
        # sort the enc_lines by input(odd lines) length
        sorted_idx = sorted(range(0, len(enc_lines), 2), key=lambda k: len(enc_lines[k]), reverse=True)
        sorted_enc_lines = [(enc_lines[i], enc_lines[i+1]) for i in sorted_idx]
        # save to pickle file
        _, file_name = os.path.split(f)
        bn = os.path.splitext(file_name)[0]
        with open(join(args.save_dir, bn+'.pkl'), 'wb') as fs:
            pickle.dump(sorted_enc_lines, fs)
        with open(join(args.save_dir, 'config.yaml'), 'w') as fs:
            yaml.dump(vars(args), fs, default_flow_style=False)

def preprocess(args):
    print("Preprocessing...")
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    sets = [d for d in os.listdir(args.data_dir)]
    voc = genVoc(args, sets)
    encode(args, sets, voc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate vocabulary and turn corpus into sequence of indices.')
    parser.add_argument('--data_dir', type=str, help='Corpus directory')
    parser.add_argument('--save_dir', type=str, help='Directory for saving generated data.')
    parser.add_argument('--size', default=10000, type=int, help='Size of the vocabulary.')
    parser.add_argument('--normalize', action='store_true', help='Normalize the strings.')
    args = parser.parse_args()
    preprocess(args)
