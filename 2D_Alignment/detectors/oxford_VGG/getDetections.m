function [detections, landmarks] = getDetections(opts, imgPath, preCalcValues)
%ALIGN_FACE run face detection over the image and returns the landmarks of
% each face.
% opts - alignment params
% imgPath - input image path
% preCalcValues - optional input with pre-calculated values : detections, landmarks
%   if landmarks is gived then only alignment is done
%   if detections is given then landmarks detection & alignment is done

    if ~exist('preCalcValues','var')
        preCalcValues = [];
    end
    if ~isfield(preCalcValues, 'detections')
      preCalcValues.detections = [];
    end
    if ~isfield(preCalcValues, 'landmarks')
      preCalcValues.landmarks = [];
    end

    [I, map] = imread(imgPath);
    X = size(I);
    if (length(X) > 3)
        % multi-frame image
        I = I(:,:,:,1);
    end
    if ~isempty(map)
        I = ind2rgb(I, map);
    end
    
    if isempty(preCalcValues.landmarks)
        if isempty(preCalcValues.detections)
            detections = runfacedet(I, imgPath);
        else
            detections = preCalcValues.detections;
        end
        nFaces = size(detections, 2);
    else
        nFaces = length(preCalcValues.landmarks);
    end
    
    % assume there is a single face in the image
    landmarks = cell(1, nFaces);
    
    for iFace = 1:nFaces
        if isempty(preCalcValues.landmarks)
            landmarks{iFace} = findparts(opts.model, I, detections(:, iFace));
        else
            landmarks{iFace} = preCalcValues.landmarks{iFace};
        end
    end
end

