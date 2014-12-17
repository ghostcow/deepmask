data_dir=../data_files/deepId_full/CASIA
for f in $data_dir/*.mat; do 
  t7_file="${f%%.mat}.t7"
  if ! [ -e "$t7_file" ]; then
    echo "Converting $f"
    th mat2torch.lua $f
  fi  
done