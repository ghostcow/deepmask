function [detection, landmarks, aligned_img] = align_face(opts, img)
%ALIGN_FACE Summary of this function goes here
%   Detailed explanation goes here
    I=imread(img);
    
    % assume there is a single face in the img
    detection=runfacedet(I,img);
    
    landmarks=findparts(opts.model,I,detection(:,1));
    
    aligned_img=alignimg(img, landmarks, opts.alignparams);
end

