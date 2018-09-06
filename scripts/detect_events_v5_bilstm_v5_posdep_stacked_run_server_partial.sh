#!/usr/bin/env bash

run_name=${run_name}_to_${incl_tokemb_out}_pi_${incl_posemb_in}_po_${incl_posemb_out}_di_${incl_depemb_in}_do_${incl_depemb_out}
output_dir=output/${run_name}
model_dir=saved_models/${run_name}

output_proc_data_json=output_proc_data_${run_name}.json.txt
output_submission_file=output_submission_${run_name}.tbf.txt
# -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file}


cmd=train
# clear model folders
mkdir -p ${output_dir}
rm -rf -- ${model_dir}
mkdir -p ${model_dir}

python_exec=venv/bin/python
echo ${python_exec} tac_kbp/detect_events_v5_bilstm_v5_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -learning_rate_fixed:${learning_rate_fixed} -incl_tokemb_out:${incl_tokemb_out} -incl_posemb_in:${incl_posemb_in} -incl_posemb_out:${incl_posemb_out} -incl_depemb_in:${incl_depemb_in} -incl_depemb_out:${incl_depemb_out} -lstm_layers:${lstm_layers} -learning_rate_divideby:${learning_rate_divideby} -learning_rate_decr_fromepoch:${learning_rate_decr_fromepoch} -event_types_labels_file:${event_types_labels_file}
${python_exec} tac_kbp/detect_events_v5_bilstm_v5_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -learning_rate:${learning_rate} -train_epochs_cnt:${train_epochs_cnt} -emb_train:${emb_train} -emb_size:${emb_size} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size} -learning_rate_fixed:${learning_rate_fixed} -incl_tokemb_out:${incl_tokemb_out} -incl_posemb_in:${incl_posemb_in} -incl_posemb_out:${incl_posemb_out} -incl_depemb_in:${incl_depemb_in} -incl_depemb_out:${incl_depemb_out} -lstm_layers:${lstm_layers} -learning_rate_divideby:${learning_rate_divideby} -learning_rate_decr_fromepoch:${learning_rate_decr_fromepoch} -event_types_labels_file:${event_types_labels_file}

cmd=eval
echo ${python_exec} tac_kbp/detect_events_v5_bilstm_v5_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file} -event_types_labels_file:${event_types_labels_file}
${python_exec} tac_kbp/detect_events_v5_bilstm_v5_posdep_stacked.py -cmd:${cmd} -run_name:${run_name} -emb_model_type:${emb_model_type} -emb_model:${emb_model} -train_data_files:${train_data_files} -dev_data_files:${dev_data_files} -model_dir:${model_dir} -word2vec_load_bin:${word2vec_load_bin} -test_data_files:${test_data_files} -output_dir:${output_dir} -max_sents:${max_sents} -batch_size:${batch_size} -lstm_hidden_size:${lstm_hidden_size} -deps_emb_size:${deps_emb_size} -pos_emb_size:${pos_emb_size}  -output_proc_data_json:${output_proc_data_json} -output_submission_file:${output_submission_file} -event_types_labels_file:${event_types_labels_file}

