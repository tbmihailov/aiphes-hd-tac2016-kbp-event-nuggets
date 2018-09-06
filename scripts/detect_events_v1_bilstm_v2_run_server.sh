#!/usr/bin/env bash

run_name="run_v1_tr201415_eval2015"

scale_features=True

# resources
emb_model_type=w2v
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin

# word2vec_load_bin=False
word2vec_load_bin=True # for google pretrained embeddings

learning_rate=0.01
train_epochs_cnt=100
batch_size=100
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size

# set max sents to small number for debug
max_sents=0

data_dir=data

data_tac2014_train=${data_dir}/clear_data/data_tac2014_train.json
data_tac2014_eval=${data_dir}/clear_data/data_tac2014_eval.json
data_tac2015_train=${data_dir}/clear_data/data_tac2015_train.json
data_tac2015_eval=${data_dir}/clear_data/data_tac2015_eval.json

train_data_files="${data_tac2014_train};${data_tac2014_eval};${data_tac2015_train}"
dev_data_files=${data_tac2015_eval}
test_data_files=${data_tac2015_eval}


cmd=train
# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}


python tac_kbp/detect_events_v1_bilstm_v2.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size}

cmd=eval
python tac_kbp/detect_events_v1_bilstm_v2.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size}


#nohup bash detect_events_v1_bilstm_v2_run_server.sh > run_2016_09_22_v1_tr201415_eval2015_w2vgoog.out 2> run_2016_09_22_v1_tr201415_eval2015_w2vgoog.err < /dev/null &