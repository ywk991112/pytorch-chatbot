import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

from rouge import rouge_n, rouge_l

class Evaluator:
    def __init__(self, writer):
        self.count = 0
        self.total_score = 0.0
        self.writer = writer

    def cal(self, decoder_output, target_seq):
        raise NotImplementError

    @property
    def score(self):
        score = self.total_score / self.count
        self.count = 0
        self.total_score = 0.0
        return score

    def log(self, mode, iteration):
        self.writer.add_scalars(self.name, {mode: self.score}, iteration)

class Perplexity(Evaluator):
    def __init__(self, writer):
        super().__init__(writer)
        self.name = 'Perplexity'

    def cal(self, decoder_output, target_seq):
        loss = F.cross_entropy(decoder_output.permute(0, 2, 1), target_seq, ignore_index=2)
        self.total_score += torch.exp(loss).item()
        self.count += decoder_output.size(1)

class ROUGE_N(Evaluator):
    def __init__(self, writer, n):
        super().__init__(writer)
        self.n = n
        self.name = 'ROUGE_{}'.format(n)

    def cal(self, decoder_output, target_seq):
        _, pred_seq = torch.max(decoder_output.permute(1, 0, 2), -1)
        pred_seq = pred_seq.cpu()
        target_seq = target_seq.permute(1, 0).cpu()
        if self.n == 'L':
            rouge_scores = Parallel(n_jobs=-1)(delayed(rouge_l)([p], [r]) for (p, r) in zip(pred_seq, target_seq))
        elif isinstance(self.n, int):
            rouge_scores = Parallel(n_jobs=-1)(delayed(rouge_n)(p, r, self.n) for (p, r) in zip(pred_seq, target_seq))
        self.count += len(rouge_scores)
        self.total_score += sum(rouge_scores)

def get_evaluator(writer, evaluator_name):
    if evaluator_name == 'perplexity':
        return Perplexity(writer)
    elif evaluator_name == 'rouge_1':
        return ROUGE_N(writer, 1)
    elif evaluator_name == 'rouge_2':
        return ROUGE_N(writer, 2)
    elif evaluator_name == 'rouge_3':
        return ROUGE_N(writer, 3)
    elif evaluator_name == 'rouge_l':
        return ROUGE_N(writer, 'L')
    else:
        raise ValueError('Evaluator {} is not implemented'.format(evaluator_name))
