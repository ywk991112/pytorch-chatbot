# pytorch-chatbot
This is a pytorch seq2seq tutorial for [Formosa Speech Grand Challenge](https://fgc.stpi.narl.org.tw/activity/techai), which is modified from [pratical-pytorch seq2seq-translation-batched](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb).  
Here is the [tutorial](https://fgc.stpi.narl.org.tw/activity/videoDetail/4b1141305df38a7c015e194f22f8015b) in Chinese.

## Update
A new version is already implemented in branch "dev".

## Requirement
* python 3.5+
* pytorch 0.4.0
* tqdm

## Get started
#### Clone the repository
```
git clone https://github.com/ywk991112/pytorch-chatbot
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
pytorch-chatbot/data/<corpus file name>
```
Otherwise, the corpus file will be tracked by git.
#### Pretrained Model
The pretrained model on [movie\_subtitles corpus](https://www.space.ntu.edu.tw/navigate/s/229EDD285D994B82B72CEDE5B5CA0CE0QQY) with an bidirectional rnn layer and
hidden size 512 can be downloaded in [this link](https://www.space.ntu.edu.tw/navigate/s/D287C8C95A0B4877B8666A45D5D318C0QQY).
The pretrained model file should be placed in directory as followed.
```
mkdir -p save/model/movie_subtitles/1-1_512
mv 50000_backup_bidir_model.tar save/model/movie_subtitles/1-1_512
```
#### Training
Run this command to start training, change the argument values in your own need.
```
python main.py -tr <CORPUS_FILE_PATH> -la 1 -hi 512 -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
```
Continue training with saved model.
```
python main.py -tr <CORPUS_FILE_PATH> -l <MODEL_FILE_PATH> -lr 0.0001 -it 50000 -b 64 -p 500 -s 1000
```
For more options,
```
python main.py -h
```
#### Testing
Models will be saved in `pytorch-chatbot/save/model` while training, and this can be changed in `config.py`.  
Evaluate the saved model with input sequences in the corpus.
```
python main.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH>
```
Test the model with input sequence manually.
```
python main.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -i
```
Beam search with size k.
```
python main.py -te <MODEL_FILE_PATH> -c <CORPUS_FILE_PATH> -be k [-i] 
```
