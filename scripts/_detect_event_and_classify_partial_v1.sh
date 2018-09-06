#!/usr/bin/env bash


# Expects following params:
# submission_version=001
# output_dir=output
# data_dir=data
# data_tac2015_eval=${data_dir}/clear_data/data_tac2015_eval.json
#
# test_data_files=${data_tac2015_eval}

###### DETECT EVENTS
echo "###############################"
echo "######### FULL EVALUATION ##########"
echo "###############################"
#run_name="run_v1_tr201415_eval2015"

echo "output_proc_data_json=${output_proc_data_json}"

#output_proc_data_json=output_proc_data_${run_name}.json.txt
#output_submission_file=output_submission_${run_name}.tbf.txt
#
#cmd=eval
#model_dir=saved_models/${run_name}
#python tac_kbp/detect_events_v4_bilstm_v4_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

###### REALIS
run_name_r=${run_name_realis}
# "run_classify_realis_v1_tr201415_eval2015"
echo "Realis run_name_r=${run_name_r}"
model_dir=saved_models/${run_name_r}

test_data_files=${output_proc_data_json}
input_is_proc_data=True

emb_model_type=w2v
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin
emb_size=300 # used in case of emb_model_type=rand
word2vec_load_bin=True # for google pretrained embeddings

event_type_classify=False
event_realis_classify=True
event_coref=False

output_proc_data_json=output_proc_data_${run_name_r}.json.txt
output_submission_file=output_submission_${run_name_r}.tbf.txt

cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name_r} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

###### TYPE
run_name_t=${run_name_type} # "run_classify_type_v1_tr201415_eval2015"
echo "Type run_name_t=${run_name_t}"
model_dir=saved_models/${run_name_t}

test_data_files=${output_proc_data_json}
input_is_proc_data=True

emb_model_type=w2v
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin
emb_size=300 # used in case of emb_model_type=rand
word2vec_load_bin=True # for google pretrained embeddings

event_type_classify=True
event_realis_classify=False
event_coref=False

output_proc_data_json=output_proc_data_${run_name_t}.json.txt
output_submission_file=sv_${submission_version}_output_submission_${run_name_t}.tbf.txt

cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name_t} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

. _evaluate_results_v1.sh
