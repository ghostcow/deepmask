# example : ./train_deepid_patch.sh [patch_index]
#	    patch_index = 1-5
patch_index=$1
data_path=../../data_files/deepId/CFW_PubFig_SUFR/CFW_PubFig_SUFR.t7
output_dir_name=CFW_PubFig_SUFR_deepID.3.64_dropout_flipped_ReLu
model_name='model_deepID.3.64'
th doall.lua --patchIndex $patch_index --dataPath $data_path --save $output_dir_name/patch$patch_index --modelName $model_name --learningRate 0.01 --batchSize 64