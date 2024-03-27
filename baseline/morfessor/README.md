## Generating segmentations with BertTokenizer
This will train, generate morphs and run the evaluation script:
```
./run.sh <language code>
```

If it finds a model in `data/<lang>`, it will use it rather than retrain the model. Make sure to move it if you want it to be retrained.

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
distance	2.11
f_measure	43.26
precision	46.88
recall	40.16
```
