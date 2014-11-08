function [detections, landmarks, aligned_imgs] = align_face(opts, imgPath)
%ALIGN_FACE Summary of this function goes here
%   Detailed explanation goes here
    [I, map] = imread(imgPath);
    X = size(I);
    if (length(X) > 3)
        % multi-frame image
        I = I(:,:,:,1);
    end
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    
    detections = runfacedet(I, imgPath);
    nFaces = size(detections, 2);
    
    % assume there is a single face in the image
    landmarks = cell(1, nFaces);
    aligned_imgs = cell(1, nFaces);
    
    for iFace = 1:nFaces
        landmarks{iFace} = findparts(opts.model, I, detections(:, iFace));
        aligned_imgs{iFace} = alignimg(I, landmarks{iFace}, opts.alignparams);
    end
end

