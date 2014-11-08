function box = det_to_rec(detection_col)
%DET_TO_REC Summary of this function goes here
%   Detailed explanation goes here
    box = [detection_col(1) - detection_col(3) ; 
           detection_col(2) - detection_col(3) ;
           detection_col(3)*2 ;
           detection_col(3)*2];
end

