function [detections, landmarks, aligned_imgs, detections2] = align_face(opts, imgPath)
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
    detections2 = runfacedet2(I, imgPath);
    
    
    if size(detections, 2) ~= size(detections2, 2) && ...
       size(detections2, 2)~=0   
        imshow(I);
        for i=1:size(detections, 2)
            rectangle('Position', det_to_rec(detections(:,i)), 'EdgeColor', 'g');
        end
        
        for i=1:size(detections2, 2)
            rectangle('Position', det_to_rec(detections2(:,i)), 'EdgeColor', 'r');
        end
        waitforbuttonpress
        
        fprintf('found different number of faces in images: %s\n', imgPath);
    end
    
    for i=1:size(detections, 2)
        for j=1:1:size(detections, 2)
            if detections2(j,i) - detections(j,i) > 2
               fprintf('detection is different by more than 1 pixel for: %s\n', imgPath); 
            end
        end
    end
    
    nFaces = size(detections, 2);
    
    % assume there is a single face in the image
    landmarks = cell(1, nFaces);
    aligned_imgs = cell(1, nFaces);
    
    for iFace = 1:nFaces
        landmarks{iFace} = findparts(opts.model, I, detections(:, iFace));
        aligned_imgs{iFace} = alignimg(I, landmarks{iFace}, opts.alignparams);
    end
end

