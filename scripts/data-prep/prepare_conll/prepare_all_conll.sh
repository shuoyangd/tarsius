#! /bin/sh
#
# prepare_all_conll.sh
# Copyright (C) 2019 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#

MAX_PREFX_LEN=$1
OUT_DIR=$2
RIGIDITY="strict"
# for file in `ls /home/shuoyangd/projects/salience_benchmark/conll-2012/v4/data/train/data/english/annotations/*/*/*/*_auto_conll` ; do
# for file in `ls /home/shuoyangd/projects/salience_benchmark/conll-2012/v4/data/train/data/english/annotations/bn/abc/*/*_auto_conll` ; do
mkdir -p $2
for file in `ls /export/c11/shuoyangd/projects/salience_benchmark/src/scripts/prepare_conll/split_conll/*_conll.doc*` ; do
  echo "$file"
  if [[ $1 == "" ]]; then
    python prepare_conll.py --data $file --outdir $2 -r $RIGIDITY > $2/`basename $file`.unannotated.txt
  else
    python prepare_conll.py --data $file --outdir $2 --max-prefix-length $MAX_PREFX_LEN -r $RIGIDITY > $2/`basename $file`.unannotated.txt
  fi
done
