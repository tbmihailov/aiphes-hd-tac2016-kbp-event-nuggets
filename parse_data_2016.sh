#!/usr/bin/env bash

rm *.json

base_dir=$(pwd)/data
coreNlpPath=$(pwd)/corenlp/stanford-corenlp-full-2015-12-09/*


nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/training/source" -dir_ann_nugget_brat:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/training/annotation" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2014_train.json" -dir_src_out_clear:"data_tac2014_train_clear_src" > data_tac2014_train.out 2> data_tac2014_train.err < /dev/null &
nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/eval/source" -dir_ann_nugget_brat:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2014/eval/annotation" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2014_eval.json" -dir_src_out_clear:"data_tac2014_eval_clear_src" > data_tac2014_eval.out 2> data_tac2014_eval.err < /dev/null &
nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/training/source" -dir_ann_nugget_brat:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/training/bratHopperAnn" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2015_train.json" -dir_src_out_clear:"data_tac2015_train_clear_src" > data_tac2015_train.out 2> data_tac2015_train.err < /dev/null &
nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/eval/source" -dir_ann_nugget_brat:"${base_dir}/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/eval/bratHopperAnn" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2015_eval.json" -dir_src_out_clear:"data_tac2015_eval_clear_src" > data_tac2015_eval.out 2> data_tac2015_eval.err < /dev/null &

nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/LDC2016E64_TAC_KBP_2016_Evaluation_Core_Source_Corpus/data/eng/df" -dir_ann_nugget_brat:"${base_dir}/LDC2016E64_TAC_KBP_2016_Evaluation_Core_Source_Corpus/data/eng" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2016_eval_df_nogold.json" -dir_src_out_clear:"data_tac2016_eval_clear_src_df" > data_tac2016_eval_df.out 2> data_tac2016_eval_df.err < /dev/null &
nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/LDC2016E64_TAC_KBP_2016_Evaluation_Core_Source_Corpus/data/eng/nw" -dir_ann_nugget_brat:"${base_dir}/LDC2016E64_TAC_KBP_2016_Evaluation_Core_Source_Corpus/data/eng" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2016_eval_nw_nogold.json" -dir_src_out_clear:"data_tac2016_eval_clear_src_nw" > data_tac2016_eval_nw.out 2> data_tac2016_eval_nw.err < /dev/null &

#with gold
nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/LDC2016E72_TAC_KBP_2016_Eval_Core_Set_Event_Nugget_Annotation/data/eng/df/source" -dir_ann_nugget_brat:"${base_dir}/LDC2016E72_TAC_KBP_2016_Eval_Core_Set_Event_Nugget_Annotation/data/eng/df/bratHopperAnn" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2016_eval_df_gold.json" -dir_src_out_clear:"data_tac2016_eval_clear_src_df" > data_tac2016_eval_df.out 2> data_tac2016_eval_df.err < /dev/null &
nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/LDC2016E72_TAC_KBP_2016_Eval_Core_Set_Event_Nugget_Annotation/data/eng/nw/source" -dir_ann_nugget_brat:"${base_dir}/LDC2016E72_TAC_KBP_2016_Eval_Core_Set_Event_Nugget_Annotation/data/eng/nw/bratHopperAnn" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2016_eval_nw_gold.json" -dir_src_out_clear:"data_tac2016_eval_clear_src_nw" > data_tac2016_eval_nw.out 2> data_tac2016_eval_nw.err < /dev/null &

nohup python Tac2016_EventNuggets_DataUtilities.py -cmd:convert_to_json_2014 -dir_src_txt:"${base_dir}/LDC2016E72_TAC_KBP_2016_Eval_Core_Set_Event_Nugget_Annotation/data/eng/nwfd/source" -dir_ann_nugget_brat:"${base_dir}/LDC2016E72_TAC_KBP_2016_Eval_Core_Set_Event_Nugget_Annotation/data/eng/nwfd/bratHopperAnn" -coreNlpPath:${coreNlpPath} -output_file:"data_tac2016_eval_eng_nwdf_gold.json" -dir_src_out_clear:"data_tac2016_eval_eng_nwdf_gold" > data_tac2016_eval_eng_nwdf_gold.out 2> data_tac2016_eval_eng_nwdf_gold.err < /dev/null &

######################################################
# Copy data from LAST to cluster
ssh last
cd research/TAC2016/tac2016-kbp-event-nuggets
cp *.json data/clear_data/
scp data/clear_data/*.json mihaylov@cluster.cl.uni-heidelberg.de:~/research/tac2016-kbp-event-nuggets/data/clear_data/


#############################
# Copy from last to current machine
scp mihaylov@last.cl.uni-heidelberg.de:~/research/TAC2016/tac2016-kbp-event-nuggets/data/clear_data/*.json data/clear_data/

cp data/clear_data/*.json /media/sf_Programming/TAC2016/clear_data



#########################
#####COPY GOLD DATA######
#########################
# run this on local machine

base_dir=$(pwd)/data
cp ${base_dir}/LDC2016E72_TAC_KBP_2016_Eval_Core_Set_Event_Nugget_Annotation/tbf/2016coreENG.tbf ${base_dir}/clear_data/data_tac2015_eval.tbf
cp ${base_dir}/_LDC2016E67_TAC_KBP_2015_Eval_Data_Character_Format/TAC_KBP_2015_EvalEventHopper_char.tbf ${base_dir}/clear_data/data_tac2015_eval.tbf

scp ${base_dir}/clear_data/*.tbf mihaylov@last.cl.uni-heidelberg.de:~/research/TAC2016/tac2016-kbp-event-nuggets/data/clear_data/
scp ${base_dir}/clear_data/*.tbf mihaylov@cluster.cl.uni-heidelberg.de:~/research/tac2016-kbp-event-nuggets/data/clear_data/



