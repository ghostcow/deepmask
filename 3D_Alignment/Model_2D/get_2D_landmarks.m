function [samples, paths] = get_2D_landmarks()
    load landmarks_lfw.mat;
    
    % filter out non frontal faces
    bs = [landmarks.bs];
    paths = {landmarks.path};
    
    % TODO: currently using only frontal pose
    frontal_samples = bs([bs.c]==7);
    paths = paths([bs.c]==7);
    
    % parse detected landmarks
    no_samples = size(frontal_samples,2); 
    samples = zeros(136, no_samples);
    for i=1:no_samples
        samples(:,i) = parse_detector_results(frontal_samples(i).xy);
    end
    samples = samples';
    samples = samples*2; % the ladnmarks were produced from the images resized by 1/2
end

