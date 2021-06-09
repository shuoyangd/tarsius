#! /bin/sh
#
# run_awd.sh
# Copyright (C) 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#

repo_base="/path/to/repo"
awd_lstm="$repo_base/awd-lstm-lm"
in="$repo_base/data/PTB/all"  # could also be Syneval, CoNLL or Winobias
model="$repo_base/models/lstm.number.pt"
wikitext_103="$repo_base/data/wikitext-103"
salience="vanilla"  # could also be {smoothed, integral, li, li_smoothed}
out="out"  # output directory, wherever you like

python $awd_lstm/compute_salience_filter_set.py --data-prefix $in --save $model --salience-type $salience --output $out --dict-data $wikitext_103 --batch-size 1 --cuda
