#!/usr/bin/env bash

# submit long job
# qsub -l has_gpu=YES -q gpu_long.q -b y bash detect_events_v2_bilstm_v3_posdep_run_server_hs600_dep50_pos50_gpu.sh
# qlogin -l has_gpu=YES,h_rt=28800 -q gpu_short.q -now n
# qlogin -l has_gpu=YES -q gpu_long.q -now n

# Set CUDA global variables
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-7.5/bin/:$PATH

runnh=run_et38_class_type_v2_tr1415_e2015
echo ${runnh}
nohup bash classify_eventtype_and_realis_v1_baseline_run_type_server.sh ${runnh}> ${runnh}.out 2> ${runnh}.err < /dev/null &

