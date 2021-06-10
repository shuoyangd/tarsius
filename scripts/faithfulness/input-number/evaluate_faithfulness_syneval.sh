#! /bin/sh
#
# run_awd.sh
# Copyright (C) 2021 Shuoyang Ding <shuoyangd@gmail.com>
#
# Distributed under terms of the MIT license.
#

repo_base="/path/to/repo"
neglect_thresh="0.01"
verum_dir="sa-verum"
malum_dir="sa-malum"

python split_syneval_saliency.py $repo_base/scripts/faithfulness/example-number/sota/out.verum syneval_section_split_list
mkdir -p $verum_dir
cat obj_rel_across_*pickle.sing_*plur* > $verum_dir/obj_rel_across.plur_sing.sa
cat obj_rel_across_*pickle.plur_*sing* > $verum_dir/obj_rel_across.sing_plur.sa
cat obj_rel_no_comp_across_*pickle.sing_*plur* > $verum_dir/obj_rel_no_comp_across.plur_sing.sa
cat obj_rel_no_comp_across_*pickle.plur_*sing* > $verum_dir/obj_rel_no_comp_across.sing_plur.sa
cat obj_rel_within_*pickle.sing_*plur* > $verum_dir/obj_rel_within.plur_sing.sa
cat obj_rel_within_*pickle.plur_*sing* > $verum_dir/obj_rel_within.sing_plur.sa
cat obj_rel_no_comp_within_*pickle.sing_*plur* > $verum_dir/obj_rel_no_comp_within.plur_sing.sa
cat obj_rel_no_comp_within_*pickle.plur_*sing* > $verum_dir/obj_rel_no_comp_within.sing_plur.sa
mv sent_comp*.sa $verum_dir
mv subj_rel*.sa $verum_dir
rm *.sa
python evaluate_faithfulness_syneval.py $verum_dir $neglect_thresh

python split_syneval_saliency.py $repo_base/scripts/faithfulness/example-number/sota/out.malum syneval_section_split_list
mkdir -p $malum_dir
cat obj_rel_across_*pickle.sing_*plur* > $malum_dir/obj_rel_across.plur_sing.sa
cat obj_rel_across_*pickle.plur_*sing* > $malum_dir/obj_rel_across.sing_plur.sa
cat obj_rel_no_comp_across_*pickle.sing_*plur* > $malum_dir/obj_rel_no_comp_across.plur_sing.sa
cat obj_rel_no_comp_across_*pickle.plur_*sing* > $malum_dir/obj_rel_no_comp_across.sing_plur.sa
cat obj_rel_within_*pickle.sing_*plur* > $malum_dir/obj_rel_within.plur_sing.sa
cat obj_rel_within_*pickle.plur_*sing* > $malum_dir/obj_rel_within.sing_plur.sa
cat obj_rel_no_comp_within_*pickle.sing_*plur* > $malum_dir/obj_rel_no_comp_within.plur_sing.sa
cat obj_rel_no_comp_within_*pickle.plur_*sing* > $malum_dir/obj_rel_no_comp_within.sing_plur.sa
mv sent_comp*.sa $malum_dir
mv subj_rel*.sa $malum_dir
rm *.sa
python evaluate_faithfulness_syneval.py $malum_dir $neglect_thresh
