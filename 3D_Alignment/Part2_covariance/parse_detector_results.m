function landmarks = parse_detector_results(points)
%PARSE_DETECTOR_RESULTS Summary of this function goes here
%   Detailed explanation goes here
    x1 = points(:,1);  
    y1 = points(:,2);  
    x2 = points(:,3);  
    y2 = points(:,4);

    landmarks = [(x1+x2)/2,(y1+y2)/2]';
    landmarks = landmarks(:);
end

