#!/usr/bin/env bash

# resources
emb_model_type=w2v
# emb_model="resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin"
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin
word2vec_load_bin=True # for google pretrained embeddings

# emb_model_type=rand
emb_train=True

emb_size=300 # used in case of emb_model_type=rand

data_dir=data

data_tac2014_train=${data_dir}/clear_data/data_tac2014_train.json
data_tac2014_eval=${data_dir}/clear_data/data_tac2014_eval.json
data_tac2015_train=${data_dir}/clear_data/data_tac2015_train.json
data_tac2015_eval=${data_dir}/clear_data/data_tac2015_eval.json

train_data_files="${data_tac2014_train};${data_tac2015_train}"
dev_data_files=${data_tac2014_eval}

test_data_files=${data_tac2015_eval}

max_sents=0

# Common run params
deps_emb_size=50
pos_emb_size=50
train_epochs_cnt=10
learning_rate_fixed=True

base_name=run_2016_10_25_ex1
# RUN 1
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size
learning_rate=0.1
batch_size=100

run_name="${base_name}_lstm${lstm_hidden_size}_lr${learning_rate}_bs${batch_size}"

log_file=res_${run_name}_$(date +%y-%m-%d-%H-%M-%S).log
. ex1_size_lr_detect_events_v2_bilstm_v3_posdep_run_partial.sh > ${log_file}

# RUN 1
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size
learning_rate=0.01
batch_size=100

run_name="${base_name}_lstm${lstm_hidden_size}_lr${learning_rate}_bs${batch_size}"

log_file=res_${run_name}_$(date +%y-%m-%d-%H-%M-%S).log
. ex1_size_lr_detect_events_v2_bilstm_v3_posdep_run_partial.sh > ${log_file}

# RUN 1
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size
learning_rate=0.001
batch_size=100

run_name="${base_name}_lstm${lstm_hidden_size}_lr${learning_rate}_bs${batch_size}"

log_file=res_${run_name}_$(date +%y-%m-%d-%H-%M-%S).log
. ex1_size_lr_detect_events_v2_bilstm_v3_posdep_run_partial.sh > ${log_file}


# RUN 1
lstm_hidden_size=600 # bilstm hidden size is 2*lstm_hidden_size
learning_rate=0.1
batch_size=100

run_name="run_2016_10_25_ex1_lstm${lstm_hidden_size}_lr${learning_rate}_bs${batch_size}"

log_file=res_${run_name}_$(date +%y-%m-%d-%H-%M-%S).log
. ex1_size_lr_detect_events_v2_bilstm_v3_posdep_run_partial.sh > ${log_file}

# RUN 1
lstm_hidden_size=600 # bilstm hidden size is 2*lstm_hidden_size
learning_rate=0.01
batch_size=100

run_name="run_2016_10_25_ex1_lstm${lstm_hidden_size}_lr${learning_rate}_bs${batch_size}"

log_file=res_${run_name}_$(date +%y-%m-%d-%H-%M-%S).log
. ex1_size_lr_detect_events_v2_bilstm_v3_posdep_run_partial.sh > ${log_file}

# RUN 1
lstm_hidden_size=600 # bilstm hidden size is 2*lstm_hidden_size
learning_rate=0.001
batch_size=100

run_name="run_2016_10_25_ex1_lstm${lstm_hidden_size}_lr${learning_rate}_bs${batch_size}"

log_file=res_${run_name}_$(date +%y-%m-%d-%H-%M-%S).log
. ex1_size_lr_detect_events_v2_bilstm_v3_posdep_run_partial.sh > ${log_file}

