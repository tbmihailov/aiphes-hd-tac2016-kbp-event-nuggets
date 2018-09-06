#!/usr/bin/env bash

# submit long job
# qsub -l has_gpu=YES -q gpu_long.q -b y bash detect_events_v2_bilstm_v3_posdep_run_server_hs600_dep50_pos50_gpu.sh
# qlogin -l has_gpu=YES,h_rt=3600 -q gpu_short.q -now n
# qlogin -l has_gpu=YES -q gpu_long.q -now n

# Set CUDA global variables
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-7.5/bin/:$PATH

runnh=run2016_10_26_14_02:wq_lstm_lr
echo ${runnh}
nohup bash ex1_size_lr_detect_events_v2_bilstm_v3_posdep_run_server_cmd.sh > ${runnh}.out 2> ${runnh}.err < /dev/null &

