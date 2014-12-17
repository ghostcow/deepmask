# example : ./train_deepid_patch.sh [patch_index]
#	    patch_index = 1-15
patch_index=$(printf %02d $1) #$1
data_path=/home/yossibiton/face_identification_nn/deepId_full/CASIA/CASIA.t7
output_dir_name=CASIA_deepID.3.64
patch_dir=patch$patch_index
output_dir_name=$output_dir_name/$patch_dir
echo $output_dir_name

model_name='model_deepID.3.64'
th doall.lua --useDatasetChunks --numPassesPerChunk 1 --deepIdMode 2 --patchIndex $patch_index --dataPath $data_path --save $output_dir_name --modelName $model_name --learningRate 0.01 --batchSize 64
