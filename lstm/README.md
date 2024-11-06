## Training and Evaluation
This code is currently not functioning well. F-scores are around 50.

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
