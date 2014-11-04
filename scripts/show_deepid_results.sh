#!/bin/sh
clear
output_dir_name=$1
last_results=2

# iterate directories and print latest accuracies
output_dir_path=../results_deepid/$output_dir_name
for run_name in $output_dir_path/*/; do
  echo "train :"
  x=$(ls -t ${run_name}train* | head -n1)
  last_modified=$(date -r $x)
  echo "$run_name - last modified on $last_modified"
  head $x -n1
  tail $x -n$last_results
  
  echo "test :"
  x=$(ls -t ${run_name}test* | head -n1) 
  head $x -n1
  tail $x -n$last_results
  echo ''
done
exit 0