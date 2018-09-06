#!/usr/bin/env bash

run_name="run_et18_classify_type_v1_tr201415_eval2015"
output_dir=output/${run_name}


#model dir where output models are saved after train
model_dir=saved_models/${run_name}


scale_features=True

# resources
emb_model_type=w2v
emb_model="resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin"
# emb_model=resources/closed_track/word2vec_google/GoogleNews-vectors-negative300.bin


# emb_model_type=rand
emb_train=True

emb_size=50 # used in case of emb_model_type=rand

word2vec_load_bin=False
# word2vec_load_bin=True # for google pretrained embeddings

data_dir=data

data_tac2014_train=${data_dir}/clear_data/data_tac2014_train.json
data_tac2014_eval=${data_dir}/clear_data/data_tac2014_eval.json
data_tac2015_train=${data_dir}/clear_data/data_tac2015_train.json
data_tac2015_eval=${data_dir}/clear_data/data_tac2015_eval.json

train_data_files="${data_tac2014_train};${data_tac2014_eval};${data_tac2015_train}"
dev_data_files=${data_tac2015_eval}

test_data_files=${data_tac2015_eval}

event_types_labels_file=data/TAC_KBP_eval_type_2016.txt # 18 types
# event_types_labels_file=data/TAC_KBP_eval_type_2015.txt # 38 types
# -event_types_labels_file:${event_types_labels_file}


event_type_classify=True
event_realis_classify=False
event_coref=False
# -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}

# Log reg settings
tune_c=False
param_c=1.0
# -tune_c:${tune_c} -param_c:${param_c}

# NN tensorflow settings - not used here
learning_rate=1
train_epochs_cnt=6
batch_size=50
lstm_hidden_size=50 # bilstm hidden size is 2*lstm_hidden_size

deps_emb_size=50
pos_emb_size=50

max_sents=500

# features
use_dep_tokens_left=True
use_dep_tokens_right=True
use_sent_emb=True
use_event_context=True
use_tokens_emb=True
use_dep_tokens=True
# -use_dep_tokens_left:${use_dep_tokens_left} -use_dep_tokens_right:${use_dep_tokens_right} -use_sent_emb:${use_sent_emb} -use_event_context:${use_event_context} -use_tokens_emb:${use_tokens_emb}

cmd=train
# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -tune_c:${tune_c} -param_c:${param_c} -event_types_labels_file:${event_types_labels_file} -use_dep_tokens_left:${use_dep_tokens_left} -use_dep_tokens_right:${use_dep_tokens_right} -use_sent_emb:${use_sent_emb} -use_event_context:${use_event_context} -use_tokens_emb:${use_tokens_emb} -use_dep_tokens:${use_dep_tokens}

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=output_submission_${run_name}.tbf.txt

# test_data_files=output_proc_data_run_classify_v1_tr201415_eval2015.json.txt
input_is_proc_data=False
# -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}


cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}  -input_is_proc_data:${input_is_proc_data} -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file} -event_types_labels_file:${event_types_labels_file} -use_dep_tokens_left:${use_dep_tokens_left} -use_dep_tokens_right:${use_dep_tokens_right} -use_sent_emb:${use_sent_emb} -use_event_context:${use_event_context} -use_tokens_emb:${use_tokens_emb} -use_dep_tokens:${use_dep_tokens}

