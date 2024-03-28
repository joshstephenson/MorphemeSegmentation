## Training and Evaluation
To train this model use:
```
python train.py
```

All the necessary hyperparameters, as well as the target language, can be configured in `config/config.yaml`. Use the language code from the table below.

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

To segment and evaluate the model use:
```
python evaluate.py
```
output:
```
category: all
distance	0.34
f_measure	95.44
precision	95.05
recall	95.83
```
