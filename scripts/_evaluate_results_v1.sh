#!/usr/bin/env bash
scorer=./scorer_v1.7.py

# run_name_eventdetection=run_et38_class_type_v2_tr1415_e2015
if [ -n "$1" ]
then
  run_name_eventdetection=$1
fi     # $String is null.

# test_data_file_tbf=data/clear_data/data_tac2015_eval.tbf
if [ -n "$2" ]
then
  test_data_file_tbf=$2
fi     # $String is null.

# output_submission_file=output_submission_run_et38_class_type_v2_tr1415_e2015.tbf.txt
if [ -n "$3" ]
then
  output_submission_file=$3
fi     # $String is null.

tkn_dir=dummy # not necessary - works with chars now? "data/DC2016E36_TAC_KBP_English_Event_Nugget_Detection_2014-2015/data/2015/eval/tkn/"

gold_file_tbf=${test_data_file_tbf}
echo "gold_file_tbf=${gold_file_tbf}"

system_output_tbf=${output_submission_file}
echo "system_output_tbf=${system_output_tbf}"

eval_dir=_scores/${run_name_eventdetection}
rm ${eval_dir} -r
mkdir ${eval_dir}

out_file_diff=${eval_dir}/gold_sys_diff.txt

scores_out_file=${eval_dir}/${run_name_eventdetection}.scores

echo "Evaluating system A, should be a perfect system\n"
echo $scorer -g ${gold_file_tbf} -s ${system_output_tbf} -d ${out_file_diff} -o ${scores_out_file}  --eval_mode char

$scorer -g ${gold_file_tbf} -s ${system_output_tbf} -d ${out_file_diff} -o ${scores_out_file}  --eval_mode char

#${scorer} -g ${nugget_gold_tbf} -s ${sys_file} -o ${result_dir}"/eval.scores" -d ${result_dir}"/gold_sys_diff" --eval_mode char
echo "Stored score report at ${scores_out_file}"

cat ${scores_out_file} >> ${log_file}
