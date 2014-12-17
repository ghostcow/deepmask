# example : ./train_deepid_patch.sh [patch_index]
#	    patch_index = 1-15
patch_index=$(printf %02d $1) #$1
data_path=../../data_files/deepId_full/CFW_PubFig_SUFR/CFW_PubFig_SUFR.t7
output_dir_name=CFW_PubFig_SUFR_deepID.3.160_30patches
patch_dir=patch$patch_index
output_dir_name=$output_dir_name/$patch_dir
echo $output_dir_name

model_name='model_deepID.3.160'
th doall.lua --deepIdMode 2 --patchIndex $patch_index --dataPath $data_path --save $output_dir_name --modelName $model_name --learningRate 0.0001 --batchSize 64 --loadState
