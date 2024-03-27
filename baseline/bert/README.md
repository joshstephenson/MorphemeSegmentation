## Generating segmentations with BertTokenizer
This will use 2 different pretrained models depending on the input language. To run generate the segments and run the evaluation script:
```
./run.sh <language code>
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

output example for Mongolian (mon):
```
category: all
distance	7.63
f_measure	5.27
precision	3.70
recall	9.12
```
