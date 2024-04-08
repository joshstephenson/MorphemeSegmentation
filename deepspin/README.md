## Training and Evaluation
To train, segment and evaluate use:
```
./train.sh <language code>
```

The default tokenization for this project is character level. To use subword level with a vocab size of 6000, set the following environment variable:
```
./train.sh <language code> <architecture> <tokenization>
```
Example:
```
./train.sh hun lstm chars
./train.sh hun transformer subwords
# OR
./train.sh hun lstm subwords
./train.sh hun transformer chars
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
