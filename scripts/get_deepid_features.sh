echo 'producing network features'
models_dir=../results_deepid/CFW_PubFig_SUFR_deepID.3.64_15patches
output_dir=$models_dir/features_1_12

echo 'verification set'
data_path=../data_files/deepId_full/CFW_PubFig_SUFR/CFW_PubFig_SUFR_verification.mat
output_path=$output_dir/verification
th get_deepid_features.lua $output_path --dataPath $data_path --deepIdMode 2 --save $models_dir --batchSize 64

echo 'LFW pairs data'
data_path=../data_files/deepId_full/LFW/pairs.mat
output_path=$output_dir/LFW_pairs
th get_deepid_features.lua $output_path --dataPath $data_path --deepIdMode 2 --save $models_dir --batchSize 64

echo 'LFW people data'
data_path=../data_files/deepId_full/LFW/people.mat
output_path=$output_dir/LFW_people
th get_deepid_features.lua $output_path --dataPath $data_path --deepIdMode 2 --save $models_dir --batchSize 64
