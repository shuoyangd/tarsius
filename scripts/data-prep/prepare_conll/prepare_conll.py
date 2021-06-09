# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-09-28
#
# Distributed under terms of the MIT license.

import argparse
import bisect
import gender_guesser.detector as gender
import itertools
import logging
import os.path
from scipy.sparse import lil_matrix

import pdb

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# we don't consider PRP$, such as "his", "her", "its"
# nor other pronouns other than third person, such as "you", "we", "me"
MALE_PRONOUNS = ["he", "him", "himself", "his"]
FEMALE_PRONOUNS = ["she", "her", "herself", "hers"]
# EPICENE_SINGULAR_PRONOUNS = ["it", "itself", "its"]
# EPICENE_PLURAL_PRONOUNS = ["they", "them", "themselves", "their", "theirs"]
# PRONOUNS = MALE_PRONOUNS + FEMALE_PRONOUNS + EPICENE_SINGULAR_PRONOUNS + EPICENE_PLURAL_PRONOUNS
PRONOUNS = MALE_PRONOUNS + FEMALE_PRONOUNS

MAX_MENTION_LENGTH=7

gender_guesser = gender.Detector()

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data", required=True, type=str, metavar="PATH", help="path where PTB train/dev/test found")
parser.add_argument("--outdir", required=True, type=str, metavar="DIR", help="a directory where the processed plain texts will be stored")
parser.add_argument("--max-prefix-length", "-l", type=int, default=0, help="maximum length of prefx, to avoid memory overflow at test time (default=0)")
parser.add_argument("--rigidity", "-r", type=str, choices=["strict", "lax"], default="strict", help="rigidity of the test (default=strict)")


def rindex(li, item):
    return max(loc for loc, val in enumerate(li) if val == item)

"""
class SparseList(lil_matrix):
  def __init__(self, length):
    super().__init__((length, 1))

  def __setitem__(self, key, item):
    super().__setitem__((key, 0), item)

  def __getitem__(self, key):
    return super().__getitem__((key, 0))
"""


# a sparse dictionary-based list that support span indexing.
#
# the reason for using a dictionary is that we want the value
# to be a dynamically allocated list
class SparseList:
  def __init__(self):
    self.dictionary = {}

  def __setitem__(self, key, item):
    if key in self.dictionary:
      if item not in self.dictionary[key]:  # only load distinct items
        self.dictionary[key].append(item)
    else:
      self.dictionary[key] = [item]

  def __getitem__(self, key):
    if isinstance(key, slice):
      ret = []
      for individual_key in key:
        if key in self.dictionary:
            ret.append(self.dictionary[key])
    elif isinstance(key, int):
      if key in self.dictionary:
        ret = self.dictionary[key]
      else:
        ret = [0]  # XXX: this is specific to this application
    else:
      raise TypeError("Invalid argument type.")

    return ret


class Span:
  def __init__(self, start, end):
    """
    start & end are both inclusive.
    """
    self.start = start
    self.end = end

  @staticmethod
  def is_nested(span1, span2):
    """
    Returns True if either span1 is nested in span2 (span2 is larger). (Not the other way!)
    Note especially two identical spans will be considered "nested".
    """
    if span2.start <= span1.start and span2.end >= span1.end:
      return True
    return False

  @staticmethod
  def is_intersecting(span1, span2):
    """
    Returns True if either span1 is intersected with span2, but not nested.
    Note especially two identical spans will be considered "intersected".
    """
    if span2.start <= span1.start <= span2.end:
      return True
    if span2.start <= span1.end <= span2.end:
      return True
    return False

  def __sub__(self, other_span):
    """
    Returns a list of spans, depending on how the two spans overlap:
    1. if two spans are intersected but not nested, will return 1 span for which the part that are intersected will be removed;
    2. if two spans are nested and "other_span" is smaller, but there is no boundary-sharing,
        will return 1 span for which the nested span will be removed;
    3. if two spans are nested and "other_span" is smaller, but there is no boundary-sharing,
        will return 2 spans for which the nested span will be removed,
        and since there is no boundary-sharing, the larger span will be broken into 2 pieces after removal.
    4. if two spans are nested and "other_span" is larger, will return 0 spans
    5. if two spans are disjoint, will return 1 span that's identical to self.
    """
    if Span.is_intersecting(self, other_span):
      if other_span.start <= self.start <= other_span.end:
        return [Span(other_span.end+1, self.end)] if other_span.end < self.end else []
      else:
        return [Span(self.start, other_span.start-1)] if self.start < other_span.start else []
    elif Span.is_nested(self, other_span) or Span.is_nested(other_span, self):
      # case 2
      if self.start == other_span.start:
        return [Span(min(other_span.end, self.end)+1, max(other_span.end, self.end))] if self.end != other_span.end else []
      elif self.end == other_span.end:
        return [Span(min(other_span.start, self.start), max(other_span.start, self.start)-1)] if self.start != other_span.start else []
      else:
        # case 3
        if self.start < other_span.start:
          return [Span(self.start, other_span.start-1), Span(other_span.end+1, self.end)]
        # case 4
        else:
          return []
    # case 5
    else:
        return [self]

  def __lt__(self, other_span):
    if self.start < other_span.start:
      return True
    else:
      return self.end < other_span.end
    return False

  def __gt__(self, other_span):
    if self.start > other_span.start:
      return True
    else:
      return self.end > other_span.end
    return False

  def __le__(self, other_span):
    return not self.__gt__(other_span)

  def __ge__(self, other_span):
    return not self.__lt__(other_span)

  def __eq__(self, other_span):
    return self.start == other_span.start and self.end == other_span.end

  def __ne__(self, other_span):
    return not self.__eq__(other_span)

  def __str__(self):
    return "{0}, {1}".format(self.start, self.end)


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
    """
    Assuming CoNLL-2012 format, where the columns are:
    DOCID PARTID WORDID WORD POSTAG PARSE LEMMA FRAMESETID SENSE SPEAKER NE PREDARG COREF
    """
    sent = []
    row_str = self.file.readline().strip()
    while row_str != "":
      if row_str.startswith("#"):
        row_str = self.file.readline().strip()
        continue
      row = {}
      columns = row_str.split()
      row["DOCID"] = columns[0]
      row["PARTID"] = columns[1]
      row["WORDID"] = columns[2] if len(columns) > 2 else "_"
      row["WORD"] = columns[3] if len(columns) > 3 else "_"
      row["POSTAG"] = columns[4] if len(columns) > 4 else "_"
      row["PARSE"] = columns[5] if len(columns) > 5 else "_"
      row["LEMMA"] = columns[6] if len(columns) > 6 else "_"
      row["FRAMESETID"] = columns[7] if len(columns) > 7 else "_"
      row["SENSE"] = columns[8] if len(columns) > 8 else "_"
      row["SPEAKER"] = columns[9] if len(columns) > 9 else "_"
      row["NE"] = columns[10] if len(columns) > 10 else "_"
      row["PREDARG"] = "\t".join(columns[11:-1])
      row["COREF"] = columns[-1]
      sent.append(row)
      row_str = self.file.readline().strip()
    return sent

  def close(self):
    self.file.close()


def parse_coref_idx(string):
  """
  return: ([starting], [ending], [singleton])
  """
  starting = []
  ending = []
  singleton = []
  idxes = string.strip().split('|')
  for idx in idxes:
    if idx.startswith('(') and idx.endswith(')'):
      singleton.append(int(idx[1:-1]))
    elif idx.startswith('('):
      starting.append(int(idx[1:]))
    elif idx.endswith(')'):
      ending.append(int(idx[:-1]))
  return starting, ending, singleton


def extract_entities_and_pronouns(sent, entities, pronouns, genders, gwid_base=0):
  """
  gender: 0 -> female, 1 -> male, 2 -> epicene_sing, 3 -> epicene_plur

  gwid refers to "global word id", which is a single index that allows one to refer to a word in this document
  (we don't really care about sentence id for our test)
  """
  open_idxes = []
  open_gwidxes = []
  gwid = gwid_base
  for token_idx, token in enumerate(sent):
    # extract entity
    if token["COREF"] != "_":
      # a single token could be the start/end of a multi-word entity
      # or the start AND the end of a single-word entity, called singleton
      starting, ending, singleton = parse_coref_idx(token["COREF"])

      for idx in starting:
        open_idxes.append(idx)
        open_gwidxes.append(gwid)
      for idx in ending:
        if idx in entities:
            qidx = rindex(open_idxes, idx)
            entities[idx].append((open_gwidxes[qidx], gwid))
        else:
            qidx = rindex(open_idxes, idx)
            if gwid - open_gwidxes[qidx] < MAX_MENTION_LENGTH:
                entities[idx] = [(open_gwidxes[qidx], gwid)]

        # does some gender propagation outside of pronouns
        entity_tokens = []
        for tmp_gwid in range(open_gwidxes[qidx], gwid):
          entity_tokens.append(sent[token_idx-(gwid-open_gwidxes[qidx])]["WORD"])

        # case 2: Mr., Ms., Mrs.
        if not idx in genders and len(entity_tokens) < 3:
          if entity_tokens[0] == "Mr.":
            genders[idx] = 1  # male
          elif entity_tokens[0] == "Ms." or entity_tokens[0] == "Mrs.":
            genders[idx] = 2  # female

        # case 3: a special case: Washington
        if not idx in genders and entity_tokens[0] == "Washington":
          genders[idx] = 0

        # case 4: name matching
        if not idx in genders and len(entity_tokens) < 3:
          g = gender_guesser.get_gender(entity_tokens[0])
          if g == "male" or g == "mostly_male":
            genders[idx] = 1
          elif g == "female" or g == "mostly female":
            genders[idx] = 2

        # clean up the processed entity idx
        del open_idxes[qidx]
        del open_gwidxes[qidx]
      for idx in singleton:
        if idx in entities:
            entities[idx].append((gwid, gwid))
        else:
            entities[idx] = [(gwid, gwid)]

      # filter out ones that are pronouns
      if token["POSTAG"].startswith("PRP") and token["WORD"].lower() in PRONOUNS and \
          len(singleton) == 1:  # filter the case where pronoun is not associated with entities
        pronouns.append((gwid, singleton[0]))
        if token["WORD"].lower() in MALE_PRONOUNS:
          genders[singleton[0]] = 1
        elif token["WORD"].lower() in FEMALE_PRONOUNS:
          genders[singleton[0]] = 2
        # elif token["WORD"].lower() in EPICENE_SINGULAR_PRONOUNS:
        #   genders[singleton[0]] = 3
        # elif token["WORD"].lower() in EPICENE_PLURAL_PRONOUNS:
        #   genders[singleton[0]] = 4
        else:
          raise Exception("PRONOUN is not a union of all pronoun categories")

    # advance gwid
    gwid += 1

  return gwid


def filter_test_cases(entities, pronouns, eid2gender, gwid2gender):
  idx = 0
  prev_gwid = -1
  gender_set = set()
  # each entry is a test case
  while idx < len(pronouns):
    gwid, eid = pronouns[idx]
    # filter case 1: should at least cover two genders
    # if already covering more than two, no need to check any more
    # this is not necessary with case 2
    # if not (1 in gender_set and 2 in gender_set):
    #     if gwid2gender[prev_gwid+1:gwid] is None:
    #       gender_set = gender_set | set([])
    #     else:
    #       gender_set = gender_set | set(gwid2gender[prev_gwid+1:gwid].toarray().transpose().tolist()[0])
    # if not (1 in gender_set and 2 in gender_set):
    #   del pronouns[idx]
    #   prev_gwid = gwid
    #   continue

    # filter case 2: the nearest entity metion should carry a different
    # gender as the predicted pronoun

    # it is important that we query pronoun genders in this way,
    # because only this will give us the gender of a singleton entity mention (spans only one word), which is unique
    # this is especially important for pronouns that are nested with a entity mention which has a larger span
    pronoun_gender = eid2gender[eid]
    is_nearest_mention_of_different_gender = False
    for tmp_gwid in range(gwid-1, prev_gwid, -1):
      if not (len(gwid2gender[tmp_gwid]) == 1 and gwid2gender[tmp_gwid][0] == 0):  # nearest entity mention seen
        is_nearest_mention_of_different_gender = any([ gender != pronoun_gender for gender in filter(lambda x: x != 0, gwid2gender[tmp_gwid]) ])
        break

    if not is_nearest_mention_of_different_gender:
      del pronouns[idx]
      prev_gwid = gwid
      continue
    else:  # this should happen before the last filtering case
      idx += 1

    prev_gwid = gwid


def match_tokenization_style(prefx_string):
  ret = prefx_string
  ret = ret.replace("/.", ".").replace("/?", "?").replace("/-", "?")
  return ret


def remove_nesting_spans(subjs, attrs):
  subj_spans = sorted([ Span(subj[0], subj[1]) for subj in subjs ])
  attr_spans = sorted([ Span(attr[0], attr[1]) for attr in attrs ])
  problematic_subj_spans = []
  # the case where subj span is larger
  for idx, subj_span in enumerate(subj_spans):
    attr_sesarch_start_idx = bisect.bisect_left(attr_spans, Span(subj_span.start, subj_span.start))
    attr_search_end_idx = bisect.bisect_right(attr_spans, Span(subj_span.end, subj_span.end))
    potential_attrs = attr_spans[attr_sesarch_start_idx:attr_search_end_idx]
    nesting_attrs = []
    for attr_span in potential_attrs:
      if Span.is_nested(attr_span, subj_span):
        nesting_attrs.append(attr_span)
    if nesting_attrs != []:
      problematic_subj_spans.append(tuple([idx, subj_span] + nesting_attrs))

  problematic_attr_spans = []
  # the case where attr span is larger
  for idx, attr_span in enumerate(attr_spans):
    subj_sesarch_start_idx = bisect.bisect_left(subj_spans, Span(attr_span.start, attr_span.start))
    subj_search_end_idx = bisect.bisect_right(subj_spans, Span(attr_span.end, attr_span.end))
    potential_subjs = subj_spans[subj_sesarch_start_idx:subj_search_end_idx]
    nesting_subjs = []
    for subj_span in potential_subjs:
      if Span.is_nested(subj_span, attr_span):
        nesting_subjs.append(subj_span)
    if nesting_subjs != []:
      problematic_attr_spans.append(tuple([idx, attr_span] + nesting_subjs))

  for span_info in problematic_subj_spans:
    result = []
    running_subj_span = span_info[1]
    for attr_span in span_info[2:]:
      ret = running_subj_span - attr_span
      if len(ret) == 1:
        running_subj_span = ret[0]
      elif len(ret) == 2:
        result.append(ret[0])
        running_subj_span = ret[1]
    result.append(running_subj_span)

    idx = span_info[0]
    del subj_spans[idx]
    for offset, result_span in enumerate(result):
      subj_spans.insert(idx+offset, result_span)

  for span_info in problematic_attr_spans:
    result = []
    running_attr_span = span_info[1]
    for attr_span in span_info[2:]:
      ret = running_attr_span - attr_span
      if len(ret) == 1:
        running_attr_span = ret[0]
      elif len(ret) == 2:
        result.append(ret[0])
        running_attr_span = ret[1]
    result.append(running_attr_span)

    idx = span_info[0]
    del attr_spans[idx]
    for offset, result_span in enumerate(result):
      attr_spans.insert(idx+offset, result_span)

  subj_ret = [ (span.start, span.end) for span in subj_spans ]
  attr_ret = [ (span.start, span.end) for span in attr_spans ]
  return subj_ret, attr_ret


def main(options):
  conll_reader = CoNLLReader(open(options.data, 'r'))
  pronouns = []
  entities = {}
  genders = {}

  sent = conll_reader.readsent()
  sent_tokens = []
  sent_boundaries = []
  gwid = 0
  while sent != []:
    sent_tokens.extend([token["WORD"] for token in sent])
    gwid = extract_entities_and_pronouns(sent, entities, pronouns, genders, gwid)
    sent_boundaries.append(gwid)
    sent = conll_reader.readsent()

  # build a mapping from gwid to gender
  # 0 -> not annotated
  # 1 -> male
  # 2 -> female
  # 3 -> epicene singular
  # 4 -> epicene plural
  # !CAVEAT!: in times where there are overlapping spans for multiple entity mentions,
  # a single gwid might correspond to multiple gender tags

  # gwid2gender = SparseList(gwid)  # old version of SparseList
  gwid2gender = SparseList()
  for eid in entities:
    spans = entities[eid]
    if eid in genders:
      g = genders[eid]
    else:
      # print entities without annotated gender (may include epicene ones, depends on what the final decision is)
      # for span in spans:
      #   print(sent_tokens[span[0]:span[1]+1])
      g = 0
    for start, end in spans:
      for idx in range(start, end + 1):
        gwid2gender[idx] = g

  # filter happens here
  filter_test_cases(entities, pronouns, genders, gwid2gender)

  filename = os.path.basename(options.data)
  outfile_prefx = open(os.path.join(options.outdir, ".".join([filename, "prefx", "txt"])), 'w')
  outfile_tag = open(os.path.join(options.outdir, ".".join([filename, "tag", "txt"])), 'w')
  outfile_subjs = open(os.path.join(options.outdir, ".".join([filename, "subjs", "txt"])), 'w')
  prev_gwid = -1

  gender1_spans = list(itertools.chain.from_iterable(entities.get(key, []) for key in filter(lambda eid: genders[eid] == 1, genders.keys())))
  gender2_spans = list(itertools.chain.from_iterable(entities.get(key, []) for key in filter(lambda eid: genders[eid] == 2, genders.keys())))
  # each pronoun holds one test case
  for gwid, eid in pronouns:
    # decide a proper prefx boundary, heuristics are follows:
    # 1. if max_prefix_length < gwid, take the whole prefix
    # 2. else, find the closest sentence boundary that makes a prefix shorter than max_prefix_length
    # 3. the bounday may also not be shorter than the last sentence boundary
    if options.max_prefix_length < 1 or gwid < options.max_prefix_length:
      prefx = sent_tokens[:gwid]
      effctv_boundary = 0
    else:
      last_sent_boundary_before_pronoun_idx = bisect.bisect_left(sent_boundaries, gwid)
      last_sent_boundary_before_pronoun = sent_boundaries[last_sent_boundary_before_pronoun_idx-1]
      if gwid - last_sent_boundary_before_pronoun < options.max_prefix_length:
        bidx = bisect.bisect_left(sent_boundaries, gwid - options.max_prefix_length)
        prefx = sent_tokens[sent_boundaries[bidx]:gwid]
        effctv_boundary = sent_boundaries[bidx]
      else:
        prefx = sent_tokens[last_sent_boundary_before_pronoun+1:gwid]
        effctv_boundary = last_sent_boundary_before_pronoun+1

    assert eid in entities and eid in genders
    if options.rigidity == "strict":
      # strict version
      # the filtering here rules out the entities with epicene gender
      subjs = [ (mention[0] - effctv_boundary, mention[1] - effctv_boundary)
                  for mention in filter(lambda x: effctv_boundary < x[0] and x[1] < gwid, entities[eid])
              ]
      attrs_keys = list(genders.keys())  # careful -- you don't want to include epicene entities
      attrs_keys.remove(eid)

      attrs = list(filter(lambda x: effctv_boundary < x[0] and x[1] < gwid,
                  list(
                    itertools.chain.from_iterable([ entities.get(key, [])
                      for key in attrs_keys ]
                    )
                  )
                )
              )
      attrs = [ (mention[0] - effctv_boundary, mention[1] - effctv_boundary)
                  for mention in attrs
              ]
      subjs, attrs = remove_nesting_spans(subjs, attrs)
      # filter out cases where there is no preceding subject/attractor
      if len(subjs) == 0 or len(attrs) == 0:
          continue

    # lax version: any entity with the same gender will do
    elif options.rigidity == "lax":
      if genders[eid] == 1:
        subjs = [ (mention[0] - effctv_boundary, mention[1] - effctv_boundary)
                  for mention in filter(lambda x: effctv_boundary < x[0] and x[1] < gwid, gender1_spans)
                ]
        attrs = [ (mention[0] - effctv_boundary, mention[1] - effctv_boundary)
                  for mention in filter(lambda x: effctv_boundary < x[0] and x[1] < gwid, gender2_spans)
                ]
      elif genders[eid] == 2:
        subjs = [ (mention[0] - effctv_boundary, mention[1] - effctv_boundary)
                  for mention in filter(lambda x: effctv_boundary < x[0] and x[1] < gwid, gender2_spans)
                ]
        attrs = [ (mention[0] - effctv_boundary, mention[1] - effctv_boundary)
                  for mention in filter(lambda x: effctv_boundary < x[0] and x[1] < gwid, gender1_spans)
                ]
      else:
        raise NotImplementedError

      subjs, attrs = remove_nesting_spans(subjs, attrs)
      # filter out cases where there is no preceding subject/attractor
      if len(subjs) == 0 or len(attrs) == 0:
          continue

    else:
      raise NotImplementedError

    tag = genders[eid]-1

    prefx_string = match_tokenization_style(" ".join(prefx))
    outfile_prefx.write(prefx_string + "\n")
    outfile_tag.write(str(tag) + "\n")
    outfile_subjs.write(str(subjs) + " ||| " + str(attrs) + "\n")

    prev_gwid = gwid


if __name__ == "__main__":
  ret = parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      parser.parse_known_args()[1]))

  main(options)
