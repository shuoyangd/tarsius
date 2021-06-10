#! /bin/sh
#
# run_awd.sh
# Copyright (C) 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#

repo_base="/path/to/repo"
neglect_thresh="0.01"  # if saliency magnitude is too low, set to 0 to avoid noise in ranking

bash group_by_content.sh $repo_base/data/Winobias/all.prefx.txt $repo_base/scripts/faithfulness/example-gender/sota/out.verum grouped_prefixes.verum
python evaluate_faithfulness_winobias.py grouped_prefixes.verum $neglect_thresh

bash group_by_content.sh $repo_base/data/Winobias/all.prefx.txt $repo_base/scripts/faithfulness/example-gender/sota/out.malum grouped_prefixes.malum
python evaluate_faithfulness_winobias.py grouped_prefixes.malum $neglect_thresh

