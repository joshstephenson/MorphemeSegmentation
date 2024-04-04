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

Results from `generate_all_langs.sh`. Note this includes the number of epochs it took before training was stopped due to loss plateauing.
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

Hungarian example of morpheme segments from Morfessor 2.0 versus correct morphological segmentation:
|Word|Morfessor 2.0|Morphological|
|----|-------------|-------------|
|pöfetegből|pöfeteg @@ből|pöfeteg @@ből|
|pleonazmus|p @@le @@on @@az @@m @@us|pleonazmus|
|világháborúsakban|világháborús @@ak @@ban|világ @@háború @@s @@ak @@ban|
|lugashoz|lugas @@hoz|lugas @@hoz|
|Fokvárosra|Fokváros @@ra|fok @@vár @@os @@ra|
|faljátok|fal @@játok|fal @@játok|
|ászanáké|ász @@aná @@k @@é|ászana @@k @@é|
|kirakata|kirakat @@a|ki @@rak @@at @@a|
|kopoltyúkat|kopoltyú @@k @@at|kopoltyú @@k @@at|
|közösségeké|közösség @@ek @@é|köz @@ös @@ség @@ek @@é|
