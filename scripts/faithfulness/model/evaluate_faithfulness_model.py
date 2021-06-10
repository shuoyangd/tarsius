import math
import os.path
import random
import sys
import numpy as np
from scipy.stats import kendalltau, pearsonr

NEGLECT_THRESH = 0.01
EVAL_SAMPLES=1500

sa_list = ["obj_rel_across.plur_sing.sa",
  "obj_rel_across.sing_plur.sa",
  "obj_rel_no_comp_across.plur_sing.sa",
  "obj_rel_no_comp_across.sing_plur.sa",
  "obj_rel_no_comp_within.plur_sing.sa",
  "obj_rel_no_comp_within.sing_plur.sa",
  "obj_rel_within.plur_sing.sa",
  "obj_rel_within.sing_plur.sa",
  "sent_comp.pickle.plur_MS_MV_sing_BS.prefx.txt.sa",
  "sent_comp.pickle.sing_MS_MV_plur_BS.prefx.txt.sa",
  "subj_rel.pickle.plur_MS_EV_MV_sing_ES.prefx.txt.sa",
  "subj_rel.pickle.sing_MS_EV_MV_plur_ES.prefx.txt.sa"
]

# code copied from https://stackoverflow.com/questions/55244113/python-get-random-unique-n-pairs
def decode(i):
    k = math.floor((1+math.sqrt(1+8*i))/2)
    return int(k),int(i-k*(k-1)//2)

def rand_pair(n):
    return decode(random.randrange(n*(n-1)//2))

def rand_pairs(n,m):
    return [decode(i) for i in random.sample(range(n*(n-1)//2),m)]

num_invalid = 0
taus = []
saliency_file1 = open(os.path.join(sys.argv[1]))
saliencies1 = [ np.array(eval(line.strip())) for line in saliency_file1 ]
saliency_file2 = open(os.path.join(sys.argv[2]))
saliencies2 = [ np.array(eval(line.strip())) for line in saliency_file2 ]

for saliency1, saliency2 in zip(saliencies1, saliencies2):
  if len(saliency1) != len(saliency2):
    # sys.stderr.write("warning: line #{0} has different length\n".format(idx))
    num_invalid += 1
    continue
  else:
    max_magnitude1 = np.max(np.abs(saliency1))
    max_magnitude2 = np.max(np.abs(saliency2))
    saliency1[ np.abs(saliency1) < NEGLECT_THRESH * max_magnitude1 ] = 0.0  # clamp small saliency values
    saliency2[ np.abs(saliency2) < NEGLECT_THRESH * max_magnitude2 ] = 0.0  # clamp small saliency values
    # rank1 = np.argsort(np.array(saliency1))
    rank1 = np.array(saliency1)
    # rank2 = np.argsort(np.array(saliency2))
    rank2 = np.array(saliency2)
    # tau, _ = kendalltau(rank1, rank2)
    tau, _ = pearsonr(rank1, rank2)
    # print(tau)
    taus.append(tau)
saliency_file1.close()
saliency_file2.close()

# print(num_invalid)
print("{0:.3f}".format(np.mean(taus)))
