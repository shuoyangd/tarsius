# Data Format

Each directory contains three files: `all.prefx.txt`, `all.subjs.txt` and `all.tags.txt`.

+ `all.prefx.txt` is the prefix input to the language model. One example per line.
+ `all.tags.txt` is the expected case for the evaluation. One example per line. For number agreement, 0 stands for singular and 1 stands for plural. For gender agreement, 0 stands for male and 1 stands for female.
+ `all.subjs.txt` is the token index spans for the cue and attractor sets. It has two fields separated by ` ||| `, with the first field for cues and the second for attractors. Each fields is formatted as:

```
[(span1_start, span1_end), (span2_start, span2_end), ... ]
```

All the indexes refer to the position of the word in the prefix, and are 0-based.
