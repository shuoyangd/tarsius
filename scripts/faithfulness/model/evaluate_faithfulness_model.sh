#! /bin/sh
#
# run_awd.sh
# Copyright (C) 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#

repo_base="/path/to/repo"
python evaluate_faithfulness_model.py $repo_base/scripts/faithfulness/example-gender/sota/out.verum /export/c11/shuoyangd/projects/salience_benchmark/code_release/scripts/faithfulness/example-gender/distilled/out.verum

python evaluate_faithfulness_model.py $repo_base/scripts/faithfulness/example-gender/sota/out.malum /export/c11/shuoyangd/projects/salience_benchmark/code_release/scripts/faithfulness/example-gender/distilled/out.malum
