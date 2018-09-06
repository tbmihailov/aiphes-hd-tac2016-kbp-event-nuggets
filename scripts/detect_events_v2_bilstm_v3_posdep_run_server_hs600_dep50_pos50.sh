#!/usr/bin/env bash

# submit long job
# qsub -l has_gpu=YES -q gpu_long.q -b y nohup bash detect_events_v2_bilstm_v3_posdep_run_server_hs600_dep50_pos50.sh ${learning_rate} ${lstm_hidden_size} > ${runnh}.out 2> ${runnh}.err < /dev/null &
# qlogin -l has_gpu=YES,h_rt=3600 -q gpu_short.q -now n
# qlogin -l has_gpu=YES -q gpu_long.q -now n

# Set CUDA global variables
# export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda-7.5/bin/:$PATH
#
# runnh=run2016_09_29_v2_tr201415_eval2015_hs600_dep50_pos50${learning_rate}_hs${lstm_hidden_size}
# echo ${runnh}
# nohup bash detect_events_v2_bilstm_v3_posdep_run_server_hs600_dep50_pos50.sh ${learning_rate} ${lstm_hidden_size} > ${runnh}.out 2> ${runnh}.err < /dev/null &

run_name="run_v2_tr201415_eval2015_hs600_dep50_pos50"
if [ -n "$1" ]
then
  run_name=$1
fi     # $String is null.

output_dir=output/${run_name}

#model dir where output models are saved after train
model_dir=saved_models/${run_name}

scale_features=True # Not used

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

train_data_files="${data_tac2014_train};${data_tac2014_eval};${data_tac2015_train}"
dev_data_files=${data_tac2015_eval}

test_data_files=${data_tac2015_eval}

learning_rate=0.01
train_epochs_cnt=100
batch_size=200
lstm_hidden_size=600 # bilstm hidden size is 2*lstm_hidden_size

deps_emb_size=50
pos_emb_size=50

max_sents=0

cmd=train
# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

python_path=venv/bin/python
${python_path} detect_events_v2_bilstm_v3_posdep.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}

cmd=eval
${python_path} detect_events_v2_bilstm_v3_posdep.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}

