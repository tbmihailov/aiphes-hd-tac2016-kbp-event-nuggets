#!/usr/bin/env bash

run_name="run_v1_tr201415_eval2015"
output_dir=output/${run_name}


#model dir where output models are saved after train
model_dir=saved_models/${run_name}


scale_features=True

# resources
emb_model_type=w2v
emb_model="resources/external/w2v_embeddings/qatarliving_qc_size20_win10_mincnt5_rpl_skip1_phrFalse_2016_02_23.word2vec.bin"
# emb_model=resources/closed_track/word2vec_google/GoogleNews-vectors-negative300.bin

emb_model_type=rand
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

learning_rate=0.1
train_epochs_cnt=6
batch_size=50
lstm_hidden_size=50 # bilstm hidden size is 2*lstm_hidden_size


max_sents=1000
cmd=train
# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

python tac_kbp/detect_events_v1_bilstm_v2.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size}

cmd=eval
python tac_kbp/detect_events_v1_bilstm_v2.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size}

