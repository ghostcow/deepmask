models_dir=../results_deepid/CFW_PubFig_SUFR_deepID.3.64_15patches
data_path=../data_files/deepId_full/CFW_PubFig_SUFR/CFW_PubFig_SUFR_verification.mat
output_path=$models_dir/verification

th get_deepid_features.lua $output_path --dataPath $data_path --deepIdMode 2 --save $models_dir --batchSize 64