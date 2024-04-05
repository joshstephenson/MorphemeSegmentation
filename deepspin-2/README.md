### Training and Evaluating
To train, segment and evaluate this model use:
```
./train.sh <language code>

```
The default tokenization for this project is character level. To use subword level with a vocab size of 6000, set the following environment variable:
```
SUBWORDS=1 ./train.sh <language code>
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

Configuration variables can be tweaked in `train.conf`.

output:
```
category: all
distance	0.34
f_measure	95.44
precision	95.05
recall	95.83
```
