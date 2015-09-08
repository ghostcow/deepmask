function [detections, landmarks, aligned_imgs] = align_face(opts, imgPath, preCalcValues)
%ALIGN_FACE run face detection over the image and produce aligned image of
% each face
% opts - alignment params
% imgPath - input image path
% preCalcValues - optional input with pre-calculated values : detections, landmarks
%   if landmarks is gived then only alignment is done
%   if detections is given then landmarks detection & alignment is done

    if ~exist('preCalcValues','var')
        [detections, landmarks] = getDetections(opts, imgPath);
    else
        [detections, landmarks] = getDetections(opts, imgPath, preCalcValues);
    end
    
    nFaces = size(landmarks,2);
    aligned_imgs = cell(1, nFaces);
    
    [I, map] = imread(imgPath);
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    
    for iFace = 1:nFaces
        aligned_imgs{iFace} = alignimg(I, landmarks{iFace}, opts.alignparams);
    end
end

