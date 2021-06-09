# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-05-06
#
# Distributed under terms of the MIT license.

import argparse
import logging
import math
import mlconjug
import os
import pdb
from utils.io import CoNLLReader
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--data", required=True, type=str, metavar="PATH", help="path where PTB train/dev/test found")
opt_parser.add_argument("--outdir", required=True, type=str, metavar="DIR", help="a directory where the processed plain texts will be stored")

lemmatizer = WordNetLemmatizer()
conjugator = mlconjug.Conjugator(language='en')
be_adversarials = {"is": "are", "are": "is", "was": "were", "were": "was"}


def to_text(sent):
  return " ".join([row["FORM"] for row in sent])


def to_tokens(sent):
  return [row["FORM"] for row in sent]


def adversarial(token):
  if token == "VBZ":
    return "VBP"
  elif token == "VBP":
    return "VBZ"
  else:
    raise Exception("I don't know the adversarial for this word")

# def adversarial(token):
#   if token in be_adversarials:
#     return be_adversarials[token]
#   lemma = lemmatizer.lemmatize(token, wordnet.VERB)
#   if lemma == token:
#     return conjugator.conjugate(lemma).conjug_info['indicative']['indicative present']['3s']
#   else:
#     return lemma


def filter_examples(sent, examples):
  """
  filter out some examples, listed as follows:

  (1) attractor is a modifier in a plural noun phrase, e.g. "customer services"
  (2) verb immediately follows subject (hence attractor effect would be very weak)
  (3) attractor occurs much earlier than subject (same as (3))
  """
  i = 0
  while i < len(examples):
    # (1)
    for idx, attr_id in enumerate(examples[i][2:]):
      attr_head = int(sent[attr_id]["HEAD"]) - 1
      if sent[examples[i][1]]["DEPREL"] == "nn" and \
          (sent[attr_head]["UPOSTAG"] == "NNS" or sent[attr_head]["UPOSTAG"] == "NNPS"):
        del examples[i][2+idx]  # delete this attractor
    # if there are not more attractors, delete this example
    if len(examples[i]) < 3:
      del examples[i]
    # (2)
    elif examples[i][1] - examples[i][0] == 1:
      del examples[i]
    # (3)
    elif min([abs(attr_id - examples[i][0]) for attr_id in examples[i][2:]]) > 10:
      del examples[i]
    else:
      i += 1
  return examples


def find_subj_verb_pair(sent):
  """
  find subject-verb pair in a parsed sentence with heuristics

  a pair (actually, more of a triplet) will be filtered out only if:
  (1) there are both singular and plural nouns in the sentence
  (2) one of the aforementioned nouns has a subject arc to a verb (or an adjective)
  (3) in the case of a verb, the aforementioned verb is a present tense verb
  (4) in the case of an adjective, the copula connected to the adjective is present tense
  (5) when multiple erraneous attractors (i.e. with non-agreeing numbers) present, we start by using the subject that's closest to the verb of interest, and if that's not possible, we use any noun that's closest to the verb of interest
  """

  ret = []

  # collect relevant elements
  nn_list = []
  nns_list = []
  nsubj_list = []
  cop_list = []
  for idx, row in enumerate(sent):
    if row["UPOSTAG"] == "NN" or row["UPOSTAG"] == "NNP":
      nn_list.append(row["ID"])
    if row["UPOSTAG"] == "NNS" or row["UPOSTAG"] == "NNPS":
      nns_list.append(row["ID"])
    if row["DEPREL"] == "nsubj":
      nsubj_list.append(row["ID"])
    if row["DEPREL"] == "cop" and \
        row["UPOSTAG"] != "VB" and \
        row["UPOSTAG"] != "VBN" and \
        row["UPOSTAG"] != "VBG":
      cop_list.append(row["ID"])

  if not nn_list or not nns_list or not nsubj_list:
    return []

  nsubj_head_list = [ int(sent[nsubj_id-1]["HEAD"]) for nsubj_id in nsubj_list ]

  # finding the subject that's connected to the copula
  cop_subj_list = []
  new_cop_list = []  # list of copula that has connected subject
  for cop_id in cop_list:
    cop_head = sent[cop_id-1]["HEAD"]
    if cop_head not in nsubj_head_list:
      continue
    # e.g. "are you ...?"
    if cop_head > cop_id:
      continue
    new_cop_list.append(cop_id)
    nsubj_list_index = nsubj_head_list.index(cop_head)
    cop_subj_list.append(nsubj_list_index)

  # sentence has copula
  for idx, cop_id in enumerate(new_cop_list):
    cop_subj = cop_subj_list[idx]
    if sent[cop_subj-1]["UPOSTAG"] == "NN":
      nns_candidates = [idx-1 for idx in list(filter(lambda x: x < cop_id and x != cop_subj, nns_list))]
      if nns_candidates:
        test = [cop_subj-1, cop_id-1, 0]
        test.extend(nns_candidates)
        ret.append(tuple(test))
    elif sent[cop_subj-1]["UPOSTAG"] == "NNS":
      nn_candidates = [idx-1 for idx in list(filter(lambda x: x < cop_id and x != cop_subj, nn_list))]
      if nn_candidates:
        test = [cop_subj-1, cop_id-1, 1]
        test.extend(nn_candidates)
        ret.append(tuple(test))

  # real verb case
  for nsubj_id, nsubj_head in zip(nsubj_list, nsubj_head_list):
    # e.g. "... says the chancellor"
    if nsubj_id > nsubj_head:
        continue
    if sent[nsubj_head-1]["UPOSTAG"] == "VBP" and sent[nsubj_head-1]["DEPREL"] != "cop" and \
        (sent[nsubj_id-1]["UPOSTAG"] == "NNS" or sent[nsubj_id-1]["UPOSTAG"] == "NNPS"):
      nn_candidates = [idx-1 for idx in list(filter(lambda x: x < nsubj_head and x != nsubj_id, nn_list))]
      if nn_candidates:
        test = [nsubj_id-1, nsubj_head-1, 1]
        test.extend(nn_candidates)
        ret.append(tuple(test))
    elif sent[nsubj_head-1]["UPOSTAG"] == "VBZ" and sent[nsubj_head-1]["DEPREL"] != "cop" and \
        (sent[nsubj_id-1]["UPOSTAG"] == "NN" or sent[nsubj_id-1]["UPOSTAG"] == "NNP"):
      nns_candidates = [idx-1 for idx in list(filter(lambda x: x < nsubj_head and x != nsubj_id, nns_list))]
      if nns_candidates:
        test = [nsubj_id-1, nsubj_head-1, 0]
        test.extend(nns_candidates)
        ret.append(tuple(test))

  # process all ret
  return ret


def process_file(conllu_file, outdir):
  logging.info("processing {0}".format(conllu_file))
  os.makedirs(outdir, exist_ok=True)
  filename = os.path.basename(conllu_file)
  reader = CoNLLReader(open(conllu_file))
  outfile_prefx = open(os.path.join(outdir, ".".join([filename, "prefx", "txt"])), 'w')
  outfile_tag = open(os.path.join(outdir, ".".join([filename, "tag", "txt"])), 'w')
  outfile_subjs = open(os.path.join(outdir, ".".join([filename, "subjs", "txt"])), 'w')
  for sent in reader:
    ret = find_subj_verb_pair(sent)
    for test in filter_examples(sent, ret):
      tokens = to_tokens(sent)
      prefix = tokens[:test[1]]
      outfile_prefx.write(" ".join(prefix) + "\n")
      outfile_tag.write(str(test[2]) + "\n")
      outfile_subjs.write(str(test[0]) + " " + " ".join([str(idx) for idx in test[3:]]) + "\n")
  outfile_prefx.close()
  outfile_tag.close()
  outfile_subjs.close()


def main(options):
  process_file(options.data + "-train.conllx", options.outdir)
  process_file(options.data + "-dev.conllx", options.outdir)
  process_file(options.data + "-test.conllx", options.outdir)


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
