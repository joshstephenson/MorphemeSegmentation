## Training and Evaluation
To train this model use:
```
python train.py
```

All the necessary hyperparameters, as well as the target language, can be configured in `config/config.yaml`. Language options are {ces|eng|fra|hun|ita|lat|mon|rus|spa}.

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
