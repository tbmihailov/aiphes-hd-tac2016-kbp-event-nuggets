#!/usr/bin/env bash

submission_version=001

output_dir=output
data_dir=data
data_tac2015_eval=${data_dir}/clear_data/data_tac2015_eval.json

test_data_files=${data_tac2015_eval}

###### DETECT EVENTS

run_name="run_v1_tr201415_eval2015"

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=output_submission_${run_name}.tbf.txt

cmd=eval
model_dir=saved_models/${run_name}
python tac_kbp/detect_events_v2_bilstm_v3_posdep.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

###### REALIS
run_name="run_classify_realis_v1_tr201415_eval2015"
model_dir=saved_models/${run_name}

test_data_files=${output_proc_data_json}
input_is_proc_data=True

emb_model_type=w2v
emb_model="resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin"
emb_size=300 # used in case of emb_model_type=rand
word2vec_load_bin=False # for google pretrained embeddings

event_type_classify=False
event_realis_classify=True
event_coref=False

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=output_submission_${run_name}.tbf.txt

cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

###### TYPE
run_name="run_classify_type_v1_tr201415_eval2015"
model_dir=saved_models/${run_name}

test_data_files=${output_proc_data_json}
input_is_proc_data=True

emb_model_type=w2v
emb_model="resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin"
emb_size=300 # used in case of emb_model_type=rand
word2vec_load_bin=False # for google pretrained embeddings

event_type_classify=True
event_realis_classify=False
event_coref=False

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=sv_${submission_version}_output_submission_${run_name}.tbf.txt

cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}
