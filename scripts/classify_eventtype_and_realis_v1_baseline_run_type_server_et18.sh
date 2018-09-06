#!/usr/bin/env bash

run_name="run_et18_class_type_v2_tr1415_e2016"
if [ -n "$1" ]
then
  run_name=$1
fi     # $String is null.

output_dir=output/${run_name}


#model dir where output models are saved after train
model_dir=saved_models/${run_name}


scale_features=True

# resources
emb_model_type=w2v
# emb_model="resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin"
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin

# emb_model_type=rand
emb_train=True

emb_size=300 # used in case of emb_model_type=rand

word2vec_load_bin=True # for google pretrained embeddings

data_dir=data

data_tac2014_train=${data_dir}/clear_data/data_tac2014_train.json
data_tac2014_eval=${data_dir}/clear_data/data_tac2014_eval.json
data_tac2015_train=${data_dir}/clear_data/data_tac2015_train.json
data_tac2015_eval=${data_dir}/clear_data/data_tac2015_eval.json

data_tac2016_eval=${data_dir}/clear_data/data_tac2016_eval_eng_nwdf_gold.json

train_data_files="${data_tac2014_train};${data_tac2015_train};${data_tac2014_eval};${data_tac2015_eval}"
dev_data_files=${data_tac2016_eval}

test_data_files=${data_tac2016_eval}

event_types_labels_file=data/TAC_KBP_eval_type_2016.txt # 18 types
# event_types_labels_file=data/TAC_KBP_eval_type_2015.txt # 38 types
# -event_types_labels_file:${event_types_labels_file}


event_type_classify=True
event_realis_classify=False
event_coref=False
# -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref}

# Log reg settings
tune_c=False
param_c=0.1
# -tune_c:${tune_c} -param_c:${param_c}

# NN tensorflow settings - not used here
learning_rate=1
train_epochs_cnt=6
batch_size=50
lstm_hidden_size=50 # bilstm hidden size is 2*lstm_hidden_size

deps_emb_size=50
pos_emb_size=50

max_sents=0

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

python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref} -tune_c:${tune_c} -param_c:${param_c} -event_types_labels_file:${event_types_labels_file} -use_dep_tokens_left:${use_dep_tokens_left} -use_dep_tokens_right:${use_dep_tokens_right} -use_sent_emb:${use_sent_emb} -use_event_context:${use_event_context} -use_tokens_emb:${use_tokens_emb} -use_dep_tokens:${use_dep_tokens}

cmd=eval
python classify_eventtype_and_realis_v1_baseline.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -event_type_classify:${event_type_classify} -event_realis_classify:${event_realis_classify} -event_coref:${event_coref} -event_types_labels_file:${event_types_labels_file} -use_dep_tokens_left:${use_dep_tokens_left} -use_dep_tokens_right:${use_dep_tokens_right} -use_sent_emb:${use_sent_emb} -use_event_context:${use_event_context} -use_tokens_emb:${use_tokens_emb} -use_dep_tokens:${use_dep_tokens}

