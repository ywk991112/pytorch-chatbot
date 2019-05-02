import yaml
import pickle
from dataloader import get_loader
from preprocess import Voc

def foo(tol):
    x, y = tol
    voc = pickle.load(open('data/voc.pkl', 'rb'))
    x = [voc.index2word[z] for z in x]
    y = [voc.index2word[z] for z in y]
    print(x, y)

pairs = pickle.load(open('data/task_01_air_ticket_booking_CH_dev.pkl', 'rb'))

print(pairs[0])
print(pairs[1])
print(pairs[2])

dl = get_loader(yaml.load(open('config/test_book.yaml', 'rb')), 'valid')
dll = list(dl)

for x, y, z in dl:
    foo((x[:, 0].tolist(), y[:, 0].tolist()))
    break
