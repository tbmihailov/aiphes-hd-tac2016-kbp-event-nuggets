#!/usr/bin/env bash

#nohup bash detect_events_v1_bilstm_v2_run_server.sh > run_2016_09_22_v1_tr201415_eval2015_w2vgoog.out 2> run_2016_09_22_v1_tr201415_eval2015_w2vgoog.err < /dev/null &

learning_rate=0.01
train_epochs_cnt=100
batch_size=100
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size

emb_model_type=w2v
emb_model=resources/word2vec/GoogleNews-vectors-negative300.bin
word2vec_load_bin=True # for google pretrained embeddings

run_name="run_v1_tr201415_eval2015"

. detect_events_v1_bilstm_v2_run_server_partial.sh
