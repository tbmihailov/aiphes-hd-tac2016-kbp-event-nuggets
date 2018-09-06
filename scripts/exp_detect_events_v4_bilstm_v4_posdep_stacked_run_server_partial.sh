#!/usr/bin/env bash

echo "#######################"
echo "#####RUN PARTIAL#######"
echo "#######################"
echo "run_name=${run_name}"

output_dir=output/${run_name}
echo "output_dir=${output_dir}"

model_dir=saved_models/${run_name}
echo "model_dir=${model_dir}"

output_proc_data_json=output_proc_data_${run_name}.json.txt
echo "output_proc_data_json=${output_proc_data_json}"

output_submission_file=output_submission_${run_name}.tbf.txt
# -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}

python_exec=venv/bin/python

cmd=train
# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}
#
echo ${python_exec} tac_kbp/exp_detect_events_v4_bilstm_v4_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -learning_rate_fixed:${learning_rate_fixed} -incl_tokemb_out:${incl_tokemb_out} -incl_posemb_in:${incl_posemb_in} -incl_posemb_out:${incl_posemb_out} -incl_depemb_in:${incl_depemb_in} -incl_depemb_out:${incl_depemb_out} -lstm_layers:${lstm_layers} -learning_rate_divideby:${learning_rate_divideby} -learning_rate_decr_fromepoch:${learning_rate_decr_fromepoch} -event_types_labels_file:${event_types_labels_file}
${python_exec} tac_kbp/exp_detect_events_v4_bilstm_v4_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -learning_rate_fixed:${learning_rate_fixed} -incl_tokemb_out:${incl_tokemb_out} -incl_posemb_in:${incl_posemb_in} -incl_posemb_out:${incl_posemb_out} -incl_depemb_in:${incl_depemb_in} -incl_depemb_out:${incl_depemb_out} -lstm_layers:${lstm_layers} -learning_rate_divideby:${learning_rate_divideby} -learning_rate_decr_fromepoch:${learning_rate_decr_fromepoch} -event_types_labels_file:${event_types_labels_file}

cmd=eval
echo ${python_exec} tac_kbp/exp_detect_events_v4_bilstm_v4_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file} -event_types_labels_file:${event_types_labels_file}
${python_exec} tac_kbp/exp_detect_events_v4_bilstm_v4_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file} -event_types_labels_file:${event_types_labels_file}

#eval with realis and type
echo "Param check before _detect_event_and_classify_partial_v1.sh"
echo "run_name_eventdetection=${run_name_eventdetection}"
echo "run_name_realis=${run_name_realis}"
echo "run_name_type=${run_name_type}"

. _detect_event_and_classify_partial_v1.sh
