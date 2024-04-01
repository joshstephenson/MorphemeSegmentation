## Generating segmentations with BertTokenizer
This will use 2 different pretrained models depending on the input language. To run generate the segments and run the evaluation script:
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

Results from `generate_all_langs.sh`:
|Language|Epochs|F1 Score|
|--------|------|--------|
|Hungarian|	6|	24.00|
|Czeck|	7|	20.42|
|English|	6|	24.43|
|French|	8|	12.66|
|Italian|	6|	9.08|
|Latin|	10|	8.84|
|Mongolian|	6|	5.27|
|Russian|	8|	13.81|
|Spanish|	7|	16.57|
