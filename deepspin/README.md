## About

This allows for training and predicting morphemes on the 2022SigmorphonST task using either an LSTM or Transformer architecture, along with either a character-level tokenization or a subword tokenization (via sentencepiece).

## Training and Evaluation
To train, segment and evaluate the command arguments are:
```
./run.sh <language code> <architecture> <tokenization> <OPTIONAL: subword vocab size>
```
Example:
```
./run.sh hun lstm chars
./run.sh hun transformer subwords
# OR
./run.sh hun lstm subwords
./run.sh hun transformer chars
# OR to change subword vocab target size:
./run.sh hun lstm subwords 200 # Default will be 6000
```

Language options include:
| Language | Language code |
|----------|---------------|
|English|eng|
|French|fra|
|Hungarian|hun|
|Italian|ita|
|Latin|lat|
|Mongolian|mon|
|Russian|rus|
|Spanish|spa|

output will be something like:
```
category: all
distance	0.34
f_measure	95.44
precision	95.05
recall	95.83
```
