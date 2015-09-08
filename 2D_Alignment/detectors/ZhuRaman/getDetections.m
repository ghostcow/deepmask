function [detections, landmarks] = getDetections(opts, imgPath, preCalcValues)
%ALIGN_FACE run face detection over the image and returns the landmarks of
% each face.
% opts - alignment params
% imgPath - input image path
% preCalcValues - optional input with pre-calculated values : detections, landmarks
%   if landmarks is gived then only alignment is done
%   if detections is given then landmarks detection & alignment is done


    detections = [];
    if ~exist('preCalcValues','var')
        preCalcValues = [];
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
    
    if ~isempty(preCalcValues.landmarks)
         landmarks = preCalcValues.landmarks;
         return
    end
    
    try
        model = opts;
        bs = detect(I, model, -5);
        bs = clipboxes(I, bs);
        bs = nms_face(bs, 0.3);
    catch me
        fprintf('%s - Error - %s\n', imgPath, me.message);
        return
    end
    
    % assume only one face in the image - take the highest scoring
    x1 = bs(1).xy(:,1);
    y1 = bs(1).xy(:,2);
    x2 = bs(1).xy(:,3);
    y2 = bs(1).xy(:,4);
    ref_XY = [(x1+x2)/2,(y1+y2)/2];
    
    landmarks = cell(1, 1);
    landmarks{1} = [ref_XY(15,:); 
                    ref_XY(10,:);
                    ref_XY(21,:); 
                    ref_XY(26,:);
                    ref_XY(3,:);
                    ref_XY(1,:); 
                    ref_XY(5,:);
                    ref_XY(35,:); 
                    ref_XY(41,:);]';
                
    %imshow(I);
    %hold on;
    %plot(landmarks{1}(1,:), landmarks{1}(2,:), 'r.');
    %figure;
end

