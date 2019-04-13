import pickle
from os.path import join
import torch
from torch.utils.data import Dataset, DataLoader
import itertools

class CorpusDataset(Dataset):
    def __init__(self, data_dir, data_set, batch_size, drop_last=False):
        self.pairs = []
        for d in data_set:
            with open(join(data_dir, d+'.pkl'), 'rb') as f:
                self.pairs += pickle.load(f)
        self.bs = batch_size
        self.drop_last = drop_last

    def __len__(self):
        import math
        if self.drop_last:
            return math.ceil(len(self.pairs)/self.bs)
        else:
            return len(self.pairs)//self.bs

    def __getitem__(self, idx):
        return self.pairs[idx*self.bs: (idx+1)*self.bs] 

def collate_fn(batch):
    def pad(seqs, fillvalue=2):
        tmp = list(itertools.zip_longest(*seqs, fillvalue=fillvalue))
        return torch.LongTensor(tmp)
    inp_max_len = len(batch[0][0][0]) # length of the input sequence of the first sample
    inp_seqs, out_seqs = list(zip(*batch[0]))
    lens = [len(x) for x in inp_seqs]
    inp_seqs, out_seqs = pad(inp_seqs), pad(out_seqs)
    return inp_seqs, out_seqs, lens

def getloader(config, mode):
    bs = config['solver']['batch_size']
    if mode == 'train':
        data_set = config['solver']['train_set']
        drop_last = False
        shuffle = True
    elif mode == 'valid':
        data_set = config['solver']['valid_set']
        drop_last = False
        shuffle = False
    elif mode == 'test':
        data_set = config['solver']['test_set']
        drop_last = False
        shuffle = False
    else:
        raise NotImplementedError
    dataset = CorpusDataset(config['preprocess']['save_dir'], data_set, batch_size=bs, drop_last=drop_last)
    return DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                      shuffle=shuffle, num_workers=4)

if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Get dataloader.')
    parser.add_argument('--config', type=str, help='Config file')
    config = parser.parse_args()
    config = yaml.load(open(config.config, 'r'))
    dl = getloader(config, 'train')
