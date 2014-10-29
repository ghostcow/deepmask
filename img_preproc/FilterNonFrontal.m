clc; clear all;
addpath('face-release1.0-basic');
run('face-release1.0-basic\init.m');

mainDir = 'D:\_Dev\face_identification_nn\data_set\google\check_aligned_frontal\Benjamin Netanyahu';
images = dir(fullfile(mainDir, '*.jpg'));
for iImage = 1:length(images);
    imPath = fullfile(mainDir, images(iImage).name);
    im = imread(imPath);
    
	bs = detect(im, model, model.thresh);
    bs = clipboxes(im, bs);
    bs = nms_face(bs,0.3);

    if (~isempty(bs) && (0 == posemap(bs(1).c)))
        fprintf('%s\n', images(iImage).name);
    end
end
