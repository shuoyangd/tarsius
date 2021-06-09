# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-11-20
#
# Distributed under terms of the MIT license.

import argparse
import logging
import os

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--data", required=True, type=str, metavar="PATH", help="path where PTB train/dev/test found")
opt_parser.add_argument("--outdir", required=True, type=str, metavar="DIR", help="a directory where the processed plain texts will be stored")


class CoNLLReader:
  def __init__(self, file):
    """

    :param file: FileIO object
    """
    self.file = file

  def __iter__(self):
    return self

  def __next__(self):
    sent = self.readsent()
    if sent == []:
      raise StopIteration()
    else:
      return sent

  def readsent(self):
    sent = []
    row_str = self.file.readline().strip()
    while row_str != "":
      if row_str.startswith("#"):
        row_str = self.file.readline().strip()
        continue
      row = {}
      columns = row_str.split('\t')
      row["STR"] = columns[0]
      row["OBJ_GENDER"] = columns[1]
      row["OBJ_IDX"] = int(columns[2])
      row["OBJ_START"] = int(columns[3])
      row["REF_IDX"] = int(columns[4])
      row["REF_TYPE"] = columns[5]
      row["SUBJ_GENDER"] = columns[6]
      row["SUBJ_IDX"] = int(columns[7])
      sent.append(row)
      row_str = self.file.readline().strip()
    return sent

  def close(self):
    self.file.close()


def main(options):
  filename = os.path.basename(options.data)
  outfile_prefx = open(os.path.join(options.outdir, ".".join([filename, "prefx", "txt"])), 'w')
  outfile_tag = open(os.path.join(options.outdir, ".".join([filename, "tag", "txt"])), 'w')
  outfile_subjs = open(os.path.join(options.outdir, ".".join([filename, "subjs", "txt"])), 'w')

  data_file = open(options.data, 'r')
  data_file.readline()  # skip the header, which is not correctly commented out
  conll_reader = CoNLLReader(data_file)
  dataset = conll_reader.readsent()

  idx = 0
  assert len(dataset) > 0
  while idx < len(dataset):
    sent = dataset[idx]
    outfile_prefx.write(sent["STR"] + "\n")
    outfile_tag.write("1\n")

    if sent["OBJ_GENDER"] == "f":
      subj_idx = sent["OBJ_IDX"]
      attr_idx = sent["SUBJ_IDX"]
    else:
      attr_idx = sent["OBJ_IDX"]
      subj_idx = sent["SUBJ_IDX"]

    outfile_subjs.write("{0} ||| {1}\n".format([(subj_idx, subj_idx)], [(attr_idx, attr_idx)]))

    idx += 1

  data_file.close()
  outfile_subjs.close()
  outfile_tag.close()
  outfile_prefx.close()


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
