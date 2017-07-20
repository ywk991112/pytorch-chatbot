import torch
import re
import unicodedata

from config import MAX_LENGTH, save_dir

SOS_token = 0
EOS_token = 1
PAD_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

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

def readVocs():
    print("Reading lines...")

    # Read the file and split into lines
    import gzip
    f_zip = gzip.open('data/open_subtitles.txt.gz', 'rt')
    lines = [line for line in f_zip]

    # combine every two lines into pairs and normalize
    it = iter(lines)
    pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]

    voc = Voc('open_subtitles')
    return voc, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData():
    voc, pairs = readVocs()
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.n_words)
    torch.save(voc, '{}/training_data/{!s}.tar'.format(save_dir, 'voc'))
    torch.save(pairs, '{}/training_data/{!s}.tar'.format(save_dir, 'pairs'))
    return voc, pairs

def loadPrepareData():
    try:
        print("Start loading training data ...")
        voc = torch.load('{}/training_data/{!s}.tar'.format(save_dir, 'voc'))
        pairs = torch.load('{}/training_data/{!s}.tar'.format(save_dir, 'pairs'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData()
    return voc, pairs
	
