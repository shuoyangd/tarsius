# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-07-03
#
# Distributed under terms of the MIT license.

import pdb
import sys

file_prefix = sys.argv[1]
prefx_file = open(file_prefix + ".prefx.txt", 'w')
tag_file = open(file_prefix + ".tag.txt", 'w')

for line in sys.stdin:
  toks = [tuple(tok.split('_')) for tok in line.strip().split()]
  for idx, (word, pos) in enumerate(toks):
    prefix = [ tok[0] for tok in toks[:idx] ]
    if pos == "VBZ":
      tag = "0"
      prefx_file.write(" ".join(prefix) + "\n")
      tag_file.write(tag + "\n")
    elif pos == "VBP":
      tag = "1"
      prefx_file.write(" ".join(prefix) + "\n")
      tag_file.write(tag + "\n")
    elif word == "was":
      tag = "0"
      prefx_file.write(" ".join(prefix) + "\n")
      tag_file.write(tag + "\n")
    elif word == "were":
      tag = "1"
      prefx_file.write(" ".join(prefix) + "\n")
      tag_file.write(tag + "\n")
    else:
      pass

prefx_file.close()
tag_file.close()

