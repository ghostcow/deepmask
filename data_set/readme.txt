All datset are saved in : /data/media/datasets on pc-wolf111
Each dataset folder should have these subfolders :
	images - contain all images arranged into seperate folders for each person
	aligned - same as above, after face detection + alignment


Datasets creation log : 

* CFW - This dataset was filtered after running different face detector than our.
        Thus, we run our detection+alignment process and saved only results intersecting with 
	the pre-filtered face images (look in our code cfw/data_set/cfw/AlignDataset.m)

All others datasets were processed by first running the detection+alignment using the script 
img_preproc/AlignDataset.m.
Later they should be filtered (removing non-faces images or wrong faces)
NOTE: in case more than one face is found the additional faces are saved as "[image_name].[face_num].jpg"

* MEDS - detection+alignment. not filtered yet.
* morph - detection+alignment. not filtered yet.
* SUFR - detection+alignment. not filtered yet. (might have intersections with CFW/LFW)

* PubFig - not processed yet. (might have intersections with CFW/LFW)
* Adience - not processed yet (images should be arranged into different folders per person)

How to create dataset :
(before running the following scripts you should change some paths inside them)
1. Aligment :
run img_preproc/AlignDataset.m, after changing relevent paths in the scripts

2. Cleaning (running neral network which filter out bad samples) :
data_set/clean_aligned_faces/CreateInput.m - create a txt file with the images to process ()
data_set/clean_aligned_faces/ApplyFaceDetNetwork.lua - apply the network and produce output txt file
data_set/clean_aligned_faces/ProcessResults.m - move the images into output directory, grouped into different folders
here manual filtering is needed...
data_set/clean_aligned_faces/ArrangeFilteredResults.m - after the filtering is done run this script to arrange the images by persons

3. Create torch file :
data_set/clean_aligned_faces/SaveDatasetPaths.m
data_set/clean_aligned_faces/SaveDatasetImages.m
data_set/clean_aligned_faces/ConcatDatasetFiles.m
utils/mat2torch.lua


