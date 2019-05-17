# pytorch-chatbot
This is a pytorch seq2seq tutorial for [Formosa Speech Grand Challenge](https://fgc.stpi.narl.org.tw/activity/techai), which is modified from [pratical-pytorch seq2seq-translation-batched](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb).  
Here is the [tutorial](https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305df38a7c015e194f22f8015b) in Chinese.

## Requirement
* python 3.5+
* pytorch 0.4.0+
* tqdm
* tensorboardX
* apex(optional)

## Get started
#### Clone the repository
```
git clone https://github.com/ywk991112/pytorch-chatbot
```

#### Config
Before running through all the process below, modify the config file first.

#### Corpus
Corpus directory tree struture should be like...
```
<corpus_directory name>
├── <train.txt>
├── <valid.txt>
|       ⋮
└── <test.txt>
```

Take config file `example_config.yaml` as the example, the directory tree will be
```
/data/corpus/open_subtitles
├── opensubtitles_train.txt
├── opensubtitles_valid.txt
└── opensubtitles_test.txt
```

In each corpus file, the input-output sequence pairs should be in the adjacent lines. For example, 
```
I'll see you next time.
Sure. Bye.
How are you?
Better than ever.
```

#### Preprocessing
The corpus text files should be preprocessed first.
```
python preprocess.py --config <config_path>
```

#### Training
Run this command to start training, change the argument values in your own need.
```
python main.py --config <config_path>
```
Continue training with saved checkpoint.
```
python main.py --config <config_path> --load <checkpoint_path>
```
Run tensorboardX to see the training result.
```
tensorboard --logdir <log_path>
```
For more options,
```
python main.py -h
```

#### Testing
Evaluate the saved model with input sequences in the test corpus.
```
python main.py --config <config_path> -te
```

#### TODO
- [ ] beam search (already implemented in master branch)
- [ ] test multi gpu


