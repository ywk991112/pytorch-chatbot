# FSGC-pytorch-tutorial
This is a pytorch seq2seq tutorial for [Formosa Speech Grand Challenge](https://fgc.stpi.narl.org.tw/activity/techai), which is modified from [pratical-pytorch seq2seq-translation-batched](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb).  
Here is the [tutorial]() in Chinese.

## Get started
#### Clone the repository
```
git clone https://github.com/ywk991112/FSGC-pytorch-tutorial
```
#### Corpus
In the corpus file, the input-output sequence pairs should be in the adjacent lines. For example, 
```
I'll see you next time.
Sure. Bye.
How are you?
Better than ever.
```
The corpus files should be placed under a path like,
```
FSGC-pytorch-tutorial/data/<corpus file name>
```
Otherwise, the corpus file will be tracked by git.
#### Training
Run this command to start training, change the argument values in your own need.
```
python main.py -tr <CORPUS_FILE_PATH> -la 1 -hi 512 -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
```
For more options,
```
python main.py -h
```
#### Testing
Models will be saved in `FSGC-pytorch-tutorial/save/model` while training, and this can be changed in `config.py`.  
Evaluate the saved model with input sequences in the corpus.
```
python main.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH>
```
Test the model with input sequence manually.
```
python main.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -i
```
