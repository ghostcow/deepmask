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
* SUFR - detection+alignment. not filtered yet.

* PubFig - not processed yet.
* Adience - not processed yet (images should be arranged into different folders per person)
