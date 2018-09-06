#!/usr/bin/env bash

#nohup bash detect_events_v1_bilstm_v2_run_server.sh > run_2016_09_22_v1_tr201415_eval2015_w2vgoog.out 2> run_2016_09_22_v1_tr201415_eval2015_w2vgoog.err < /dev/null &

learning_rate=0.005
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size

runnh=run_2016_09_27_v1_tr1415_eval15_w2vgoog_lr${learning_rate}_hs${lstm_hidden_size}
echo ${runnh}
nohup bash detect_events_v1_bilstm_v2_run_server_params.sh ${learning_rate} ${lstm_hidden_size} > ${runnh}.out 2> ${runnh}.err < /dev/null &

learning_rate=0.0005
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size

runnh=run_2016_09_27_v1_tr1415_eval15_w2vgoog_lr${learning_rate}_hs${lstm_hidden_size}
echo ${runnh}
nohup bash detect_events_v1_bilstm_v2_run_server_params.sh ${learning_rate} ${lstm_hidden_size} > ${runnh}.out 2> ${runnh}.err < /dev/null &

learning_rate=0.001
lstm_hidden_size=300 # bilstm hidden size is 2*lstm_hidden_size

runnh=run_2016_09_27_v1_tr1415_eval15_w2vgoog_lr${learning_rate}_hs${lstm_hidden_size}
echo ${runnh}
nohup bash detect_events_v1_bilstm_v2_run_server_params.sh ${learning_rate} ${lstm_hidden_size} > ${runnh}.out 2> ${runnh}.err < /dev/null &
