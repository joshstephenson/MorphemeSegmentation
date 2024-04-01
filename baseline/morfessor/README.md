## Generating segmentations with Morfessor 2.0
This will train, generate morphs and run the evaluation script:
```
./generate_all_langs.sh
```

or
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

Results from `generate_all_langs.sh`:
|Language|Epochs|F1 Score|
|--------|------|--------|
|Hungarian|	6|	37.34|
|Czeck|	7|	29.79|
|English|	6|	45.14|
|French|	8|	21.37|
|Italian|	6|	13.79|
|Latin|	10|	18.10|
|Mongolian|	6|	43.26|
|Russian|	8|	10.46|
|Spanish|	7|	9.23|
