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
|Language|F1 Score|
|--------|--------|
|Hungarian|24.00|
|Czeck|20.42|
|English|24.43|
|French|12.66|
|Italian|9.08|
|Latin|8.84|
|Mongolian|5.27|
|Russian|13.81|
|Spanish|16.57|
