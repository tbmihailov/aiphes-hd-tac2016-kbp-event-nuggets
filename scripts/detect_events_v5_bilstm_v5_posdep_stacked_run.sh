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
# event_types_labels_file=
event_types_labels_file=data/TAC_KBP_eval_type_2015.txt # 38 types

learning_rate=1
learning_rate_fixed=True
learning_rate_divideby=1.2 # learning_rate=learning_rate/learning_rate_decrease
learning_rate_decr_fromepoch=3 # start decreasing learning rate after this epoch


train_epochs_cnt=3
batch_size=50
lstm_hidden_size=50 # bilstm hidden size is 2*lstm_hidden_size
lstm_layers=3

deps_emb_size=20
pos_emb_size=20

#Include embeddings in input or output layer(shortcut)
incl_tokemb_out=True
incl_posemb_in=True
incl_posemb_out=True
incl_depemb_in=True
incl_depemb_out=True

# -incl_tokemb_out:${incl_tokemb_out} -incl_posemb_in:${incl_posemb_in} -incl_posemb_out:${incl_posemb_out} -incl_depemb_in:${incl_depemb_in} -incl_depemb_out:${incl_depemb_out}
#run_name=_${run_name}_to_${incl_tokemb_out}_pi_${incl_posemb_in}_po_${incl_posemb_out}_di_${incl_depemb_in}_do_${incl_depemb_out}

max_sents=500

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=output_submission_${run_name}.tbf.txt
# -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

cmd=train
# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

python tac_kbp/detect_events_v5_bilstm_v5_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -learning_rate_fixed:${learning_rate_fixed} -incl_tokemb_out:${incl_tokemb_out} -incl_posemb_in:${incl_posemb_in} -incl_posemb_out:${incl_posemb_out} -incl_depemb_in:${incl_depemb_in} -incl_depemb_out:${incl_depemb_out} -learning_rate_divideby:${learning_rate_divideby} -learning_rate_decr_fromepoch:${learning_rate_decr_fromepoch} -event_types_labels_file:${event_types_labels_file}

cmd=eval
python tac_kbp/detect_events_v5_bilstm_v5_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file} -event_types_labels_file:${event_types_labels_file}

