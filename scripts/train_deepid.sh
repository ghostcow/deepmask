#!/bin/sh
clear
data_path=../../data_files/deepId/CFW_small2/cfw_small2.t7
output_dir_name=cfw_small2_deepID.3
model_name='model_deepID.3'

main_log_file=../../results_deepid/$output_dir_name/main.log
pids=""
for patch_index in 1 2 3 4 5
do
  echo "patch no. $patch_index"
  patch_output=$output_dir_name/patch$patch_index
  th doall.lua --dataPath $data_path --save $patch_output --modelName $model_name --patchIndex $patch_index --learningRate 0.01 --batchSize 64 &
  pids="$pids $!"
done

for job in $pids
do
  echo "PID = $job"
done

# wait for all subprocesses
wait

# we will never reach here
exit 0