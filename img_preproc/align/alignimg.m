function faceImg = alignimg(img, landmarks, alignparams)
%ALIGN_IMG Summary of this function goes here
%   Detailed explanation goes here

    % landmarks in the canonical coordinate system
    basePts = alignparams.basePts;

    % horizontal centre
    cx = mean(basePts(1, [1:4, 8:9]));

    % eye line
    top = mean(basePts(2, 1:4));

    % mouth line
    bottom = mean(basePts(2, 8:9));

    % horizontal distance between eyes
    dx = mean(basePts(1, 3:4)) - mean(basePts(1, 1:2));

    % vertical distance between eyes & mouth
    dy = bottom - top;

    % set crop region in the canonical coordinate system
    horRatio = alignparams.horRatio; 
    topRatio = alignparams.topRatio;
    bottomRatio = alignparams.bottomRatio; 

    x0 = cx - dx * horRatio;
    x1 = cx + dx * horRatio;
    y0 = top - dy * topRatio;
    y1 = bottom + dy * bottomRatio;

    % scale
    scale = alignparams.scale;

    basePts = basePts * scale;
    x0 = x0 * scale;
    x1 = x1 * scale;
    y0 = y0 * scale;
    y1 = y1 * scale;

    % compute alignment transform
    tform = cp2tform(landmarks', basePts', 'affine');
    
    % apply transform and do crop
    faceImg = imtransform(img, tform, 'bicubic', 'XData', [x0 x1], ...
                                                 'YData', [y0 y1], ...
                                                 'XYScale', 1);    
end

