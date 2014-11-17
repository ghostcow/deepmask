data_path=../data_files/LFW/people.mat
models_dir=../results_deepid/CFW_PubFig_SUFR_deepID.3.64_dropout_flipped
output_path=../results_deepid/CFW_PubFig_SUFR_deepID.3.64_dropout_flipped/deepid_LFW_people

th get_deepid_features.lua $output_path --dataPath $data_path --save $models_dir --batchSize 64