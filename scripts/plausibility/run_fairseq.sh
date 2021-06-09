#! /bin/sh
#
# run_awd.sh
# Copyright (C) 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#

repo_base="/path/to/repo"
fairseq="$repo_base/fairseq"
in="$repo_base/data/PTB/all"  # could also be Syneval, CoNLL or Winobias
model="$repo_base/models/transformer.number.pt"
wikitext_103="$repo_base/data/wikitext-103-bin"
salience="vanilla"  # could also be {smoothed, integral, li, li_smoothed}
out="out"  # output directory, wherever you like

python $fairseq/compute_salience_fairseq_set.py --data-prefix $in --path $model --context-window 2560 --softmax-batch 1024 --max-tokens 1024 --salience-type $salience --output $out --salience-batch-size 1 --cuda $wikitext_103
