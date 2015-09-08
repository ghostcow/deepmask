function faceImg = alignimg(img, landmarks, alignparams)
%ALIGN_IMG Summary of this function goes here
%   Detailed explanation goes here

    [basePts, x0, x1, y0, y1] = GetAlignedImageCoords(alignparams);
    
    % compute alignment transform
    tform = cp2tform(landmarks', basePts', 'affine');
    
    % apply transform and do crop
    img = im2double(img);
    faceImg = imtransform(img, tform, 'bicubic', 'XData', [x0 x1], ...
                                                 'YData', [y0 y1], ...
                                                 'XYScale', 1);    
end

