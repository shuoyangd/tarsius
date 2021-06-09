# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-05-02
#
# Distributed under terms of the MIT license.

import argparse
import logging
import os
import pdb
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--data", required=True, type=str, metavar="PATH", help="path where Targeted LM Eval Dataset is checked out")
opt_parser.add_argument("--outdir", required=True, type=str, metavar="DIR", help="a directory where the processed plain texts will be stored")
prefix_list = ["sent_comp", "prep", "subj_rel", "obj_rel"]
truth_is_last_prefix_list = ["sent_comp", "obj_rel_within", "obj_rel_no_comp_within"]
neighbor_verb_prefix_list = ["obj_rel_across", "obj_rel_no_comp_across"]  # these files has neighboring present tense verbs that needs to be substituted with POS tags
that_verb_prefix_list = ["subj_rel"]  # these files has present verbs following "that" that needs to be substituted with POS tags

lemmatizer = WordNetLemmatizer()
be_tags = {"is": "VBZ", "are": "VBP", "was": "VBZ", "were": "VBP"}

def easy_tag(verb):
  if verb in be_tags:
    return "1" if be_tags[verb] == "VBP" else "0"
  elif lemmatizer.lemmatize(verb, wordnet.VERB) == verb:
    return "1"
  else:
    return "0"


def substitute_verbs(toks, filename):
  if any([filename.startswith(prefix) for prefix in neighbor_verb_prefix_list]):
    verb = toks[-1]
    toks[-1] = easy_tag(verb)
  elif any([filename.startswith(prefix) for prefix in that_verb_prefix_list]):
    verb_index = toks.index("that") + 1
    toks[verb_index] = easy_tag(toks[verb_index])
  return toks


# TODO: needs a bit clean-up
# def find_all_subj(sent, limit=-1):
def find_all_subj(sent):
  """
  Find all the subjects that we would like to probe, with corpus-specific heuristic.
  We cannot use a postagger because most of these sentences in synthetic datasets are garden-path sentences.
  It is very likely that a postagger will make mistakes.

  heuristics:
    (1) all the NPs should be preceded by "the"; everything following "the" are NPs;
    (2) in case where "the" is followed by "taxi driver(s)", will use driver;
    (3) will ignore "side".
  """

  idxes = []
  for idx, word in enumerate(sent):
    if word == "the" and sent[idx+1] == "taxi" and sent[idx+2].startswith("driver"):
      idxes.append(idx+2)
    elif word == "the" and sent[idx+1] == "side":
      continue
    elif word == "the":
      idxes.append(idx+1)
  return idxes


def process_file(pickle_file, outdir):
  logging.info("processing {0}".format(pickle_file))
  os.makedirs(outdir, exist_ok=True)
  filename = os.path.basename(pickle_file)
  data_dict = pickle.load(open(pickle_file, 'rb'))
  for key in data_dict:
    # we are only interested in sentences that we can get ground truth with only the verb prediction
    # items that has two singular/plural subjects does not satisfy that condition
    if not ("sing" in key and "plur" in key):
      continue

    outfile_prefx = open(os.path.join(outdir, ".".join([filename, key, "prefx", "txt"])), 'w')
    outfile_tag = open(os.path.join(outdir, ".".join([filename, key, "tag", "txt"])), 'w')
    outfile_subjs = open(os.path.join(outdir, ".".join([filename, key, "subjs", "txt"])), 'w')
    common_prev = ""
    for verum, malum in data_dict[key]:
      verum_toks = verum.split(' ')
      malum_toks = malum.split(' ')
      common_toks = []
      for vtok, mtok in zip(verum_toks, malum_toks):
        if vtok == mtok:
          common_toks.append(vtok)
        else:
          break

      malum_subjs = find_all_subj(malum_toks)
      verum_subjs = find_all_subj(verum_toks)
      assert malum_subjs == verum_subjs
      if any([ filename.startswith(prefix) for prefix in truth_is_last_prefix_list ]):
        subjs = reversed(verum_subjs)
      else:
        subjs = verum_subjs
      subjs = [str(idx) for idx in subjs]
      # we know for sure these datasets should have two and only two subjects
      if len(subjs) != 2:
        logging.warning("potentially mallicious subjects in sentence \"{0}\"".format(verum))

      # common_toks = substitute_verbs(common_toks, filename)
      if " ".join(common_toks) == common_prev:
        continue
      else:
        common_prev = " ".join(common_toks)

      outfile_prefx.write(" ".join(common_toks) + "\n")
      outfile_tag.write(easy_tag(vtok) + "\n")
      outfile_subjs.write(" ".join(subjs) + "\n")
    outfile_prefx.close()
    outfile_tag.close()
    outfile_subjs.close()


def main(options):
  pickle_files = []
  for file in os.listdir(os.path.join(options.data, "data", "templates")):
    if any([file.startswith(prefix) for prefix in prefix_list]) and file.endswith(".pickle"):
      pickle_files.append(file)

  for pickle_file in pickle_files:
    process_file(os.path.join(options.data, "data", "templates", pickle_file), options.outdir)


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
