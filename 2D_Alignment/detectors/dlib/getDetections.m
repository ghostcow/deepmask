function [detections, landmarks] = getDetections(opts, imgPath, ~)
%ALIGN_FACE run face detection over the image and returns the landmarks of
% each face.
% opts - alignment params
% imgPath - input image path
% preCalcValues - optional input with pre-calculated values : detections, landmarks
%   if landmarks is gived then only alignment is done
%   if detections is given then landmarks detection & alignment is done
    
    % load image
    [I, map] = imread(imgPath);
    X = size(I);
    if (length(X) > 3)
        % multi-frame image
        I = I(:,:,:,1);
    end
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    
    % detect in image
    try
        ref_XY = face_detector(I, opts.predictorPath);
    catch me
        fprintf('%s - Error - %s\n', imgPath, me.message);
        return
    end
    
    landmarks = cell(1, 1);
    landmarks{1} = [ref_XY(37,:); 
                    ref_XY(40,:);
                    ref_XY(43,:); 
                    ref_XY(46,:);
                    ref_XY(32,:);
                    ref_XY(34,:); 
                    ref_XY(36,:);
                    ref_XY(49,:); 
                    ref_XY(55,:);]';
     detections = [0, 0; 250, 250];
end