import numpy as np
import sys

saliency_file = sys.argv[1]
split_file = sys.argv[2]

# read split file first
num_lines = []
field_names = []
for line in open(split_file):
  fields = line.split(',')
  num_lines.append(int(fields[0]))
  field_names.append(fields[1].strip())

num_lines = np.array(num_lines)
cum_num_lines = np.cumsum(num_lines)
current_file_idx = 0
current_file = open(field_names[current_file_idx] + ".sa", 'w')
for idx, line in enumerate(open(saliency_file)):
  if idx < cum_num_lines[current_file_idx]:
    current_file.write(line)
  else:
    current_file_idx += 1
    current_file.close()
    current_file = open(field_names[current_file_idx] + ".sa", 'w')
    current_file.write(line)

current_file.close()
