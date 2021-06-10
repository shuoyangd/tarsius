import os.path
import sys
import numpy as np
from scipy.stats import kendalltau, pearsonr

NEGLECT_THRESH = float(sys.argv[2])

sa_list = [ str(idx) + ".sa" for idx in range(147) ]

num_invalid = 0
taus = []
for saliency_file_name in sa_list:
  saliency_file = open(os.path.join(sys.argv[1], saliency_file_name))
  first_line_saliency = np.array(eval(saliency_file.readline().strip()))
  max_magnitude = np.max(np.abs(first_line_saliency))
  first_line_saliency[ np.abs(first_line_saliency) < NEGLECT_THRESH * max_magnitude ] = 0.0  # clamp small saliency values
  # standard_rank = np.argsort(first_line_saliency)
  standard_rank = first_line_saliency

  for idx, line in enumerate(saliency_file):
    saliency = np.array(eval(line.strip()))
    if len(saliency) != len(first_line_saliency):
      # sys.stderr.write("warning: line #{0} has different length\n".format(idx))
      num_invalid += 1
      continue
    else:
      max_magnitude = np.max(np.abs(saliency))
      saliency[ np.abs(saliency) < NEGLECT_THRESH * max_magnitude ] = 0.0  # clamp small saliency values
      # rank = np.argsort(np.array(saliency))
      rank = np.array(saliency)
      # tau, _ = kendalltau(rank, standard_rank)
      tau, _ = pearsonr(rank, standard_rank)
      taus.append(tau)
  saliency_file.close()

# print(num_invalid)
print("{0:.3f}".format(np.mean(taus)))
