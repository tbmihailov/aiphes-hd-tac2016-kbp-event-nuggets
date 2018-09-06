#!/usr/bin/env bash

submission_version=001

output_dir=output
data_dir=data
data_tac2015_eval=${data_dir}/clear_data/data_tac2015_eval.json

test_data_files=${data_tac2015_eval}

max_sents=0
echo ##### DETECT EVENTS #####

run_name="run_v1_tr201415_eval2015"

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=output_submission_${run_name}.tbf.txt

cmd=eval
model_dir=saved_models/${run_name}
python tac_kbp/detect_events_v2_bilstm_v3_posdep.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

echo ##### CLASSIFY REALIS #####

run_name="run_class_realis_161004_v2_tr1415_e2015"
model_dir=saved_models/${run_name}

test_data_files=${output_proc_data_json}
input_is_proc_data=True

emb_model_type=w2v
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin
emb_size=300 # used in case of emb_model_type=rand
word2vec_load_bin=True # for google pretrained embeddings

event_type_classify=False
event_realis_classify=True
event_coref=False

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=output_submission_${run_name}.tbf.txt

cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

echo ##### CLASSIFY TYPE #####

run_name="run_class_type_161004_v2_tr1415_e2015"
model_dir=saved_models/${run_name}

test_data_files=${output_proc_data_json}
input_is_proc_data=True

emb_model_type=w2v
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin
emb_size=300 # used in case of emb_model_type=rand
word2vec_load_bin=True # for google pretrained embeddings

event_type_classify=True
event_realis_classify=False
event_coref=False

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=sv_${submission_version}_output_submission_${run_name}.tbf.txt

cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}
