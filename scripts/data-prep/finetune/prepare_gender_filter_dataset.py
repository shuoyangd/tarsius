# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-10-18
#
# Distributed under terms of the MIT license.

import pdb
import sys

MALE_PRONOUNS = ["he", "him", "himself", "his"]
FEMALE_PRONOUNS = ["she", "her", "herself", "hers"]

file_prefix = sys.argv[1]
prefx_file = open(file_prefix + ".prefx.txt", 'w')
tag_file = open(file_prefix + ".tag.txt", 'w')

for line in sys.stdin:
  toks = line.strip().split()
  for idx, word in enumerate(toks):
    prefix = toks[:idx]
    if word in MALE_PRONOUNS:
      tag = "0"
      prefx_file.write(" ".join(prefix) + "\n")
      tag_file.write(tag + "\n")
    elif word in FEMALE_PRONOUNS:
      tag = "1"
      prefx_file.write(" ".join(prefix) + "\n")
      tag_file.write(tag + "\n")
    else:
      pass

prefx_file.close()
tag_file.close()

