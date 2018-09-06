#!/usr/bin/env bash

#nohup bash detect_events_v1_bilstm_v2_run_server_3_hiddensize.sh > run_2016_09_22_v1_tr201415_eval2015_w2vgoog.out 2> run_2016_09_22_v1_tr201415_eval2015_w2vgoog.err < /dev/null &
echo ==============

learning_rate=0.01
if [ -n "$1" ]
then
  learning_rate=$1
fi     # $String is null.

lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size
if [ -n "$2" ]
then
  lstm_hidden_size=$2
fi     # $String is null.

train_epochs_cnt=50
batch_size=100

emb_model_type=w2v
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin
word2vec_load_bin=True # for google pretrained embeddings

run_name="run_bilstm2_tr1415_eval15_lr${learning_rate}_hs${lstm_hidden_size}"

log_file=${run_name}_$(date +%y-%m-%d-%H-%M).log
. detect_events_v1_bilstm_v2_run_server_partial.sh > ${log_file}

