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

Example of Bert Tokenization versus correct morphological segmentation for Hungarian:
|Word|BERT|Morphological|
|----|----|---------|
|pöfetegből|p @@öfe @@teg @@ből|pöfeteg @@ből|
|pleonazmus|pl @@eon @@az @@mus|pleonazmus|
|világháborúsakban|világháború @@sak @@ban|világ @@háború @@s @@ak @@ban|
|lugashoz|lu @@gas @@hoz|lugas @@hoz|
|Fokvárosra|F @@ok @@város @@ra|fok @@vár @@os @@ra|
|faljátok|fa @@lj @@át @@ok|fal @@játok|
|ászanáké|ás @@zan @@ák @@é|ászana @@k @@é|
|kirakata|kira @@kata|ki @@rak @@at @@a|
|kopoltyúkat|kop @@olt @@y @@ú @@kat|kopoltyú @@k @@at|
|közösségeké|közös @@ségek @@é|köz @@ös @@ség @@ek @@é|

