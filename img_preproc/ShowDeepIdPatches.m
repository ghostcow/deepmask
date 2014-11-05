clear all variables; clc; close all;

%% align image 
run('init.m');
opts.alignparams.scale = 1.3; % to get 152x152 aligned image : scale = 3.5, im = aligned(78:229, 50:201, :)
opts.alignparams.horRatio = 1.7;
opts.alignparams.topRatio = 2;
opts.alignparams.bottomRatio = 1.2;

[basePts, x0, x1, y0, y1] = GetAlignedImageCoords(opts.alignparams);
landmarksTarget = bsxfun(@minus, basePts, [x0;y0]) + 1;
[detection, landmarks, aligned_imgs] = align_face(opts, 'test.jpg');
im = aligned_imgs{1};
imageDim = [size(im, 2), size(im, 1)];
figure(100); imshow(im);

%% show all patches on the aligned image
patchRadius = [15, 22, 31]; %round(15*[1, sqrt(2), 2]); % 19 / 27 / 39; 
numScales = length(patchRadius);

% first kind of patches - square
patchCenters1 = zeros(2, 5);
patchCenters1(:, 1) = 0.5*(landmarksTarget(:, 1) + landmarksTarget(:, 2)); % left eye
patchCenters1(:, 2) = 0.5*(landmarksTarget(:, 3) + landmarksTarget(:, 4)); % right eye
patchCenters1(:, 3) = 0.5*(landmarksTarget(:, 5) + landmarksTarget(:, 7)); % nose
patchCenters1(:, 4) = landmarksTarget(:, 8);                               % left lip edge
patchCenters1(:, 5) = landmarksTarget(:, 9);                               % right lip edge

% 2nd kind of patches - horizontal frames
eyesCenter = 0.5*(patchCenters1(:, 1) + patchCenters1(:, 2));
topMiddle = [eyesCenter(1); 1];
patchCenters2 = zeros(2, 4);
patchCenters2(:, 1) = 0.5*(topMiddle + eyesCenter);
patchCenters2(:, 2) = eyesCenter;
patchCenters2(:, 3) = [patchCenters2(1, 1); 0.5*(eyesCenter(2) + patchCenters1(2, 3))];
patchCenters2(:, 4) = [patchCenters2(1, 1); patchCenters1(2, 3)];
patchHalfWidths2 = [20 30 42]; % round(1.35*patchRadius)

% 2nd kind of patches - vertical frame
patchCenters3 = eyesCenter;
patchHalfWidths3 = [20 30 42];
patchHalfHeights3 = round(1.2*patchHalfWidths3);

% figure(1); imshow(im); hold on;

for iPatchType = 1:3
    if (iPatchType == 1)
        patchCenters = patchCenters1;
    elseif (iPatchType == 2)
        patchCenters = patchCenters2;
    else
        patchCenters = patchCenters3;
    end
    
    numLocs = size(patchCenters, 2);
    figure(iPatchType); hold on;
    
    for iScale = 1:numScales
        if (iPatchType == 1)
            radius = [patchRadius(iScale), patchRadius(iScale)];
        elseif (iPatchType == 2)
            radius = [patchHalfWidths2(iScale), patchRadius(iScale)];
        else
            radius = [patchHalfWidths3(iScale), patchHalfHeights3(iScale)];
        end
        
        for iCenter = 1:numLocs
           patchBorders = zeros(2, 2);
           for iCoord = 1:2
               patchBorders(1, iCoord) = max(round(patchCenters(iCoord, iCenter) - radius(iCoord)), 1);
               patchBorders(2, iCoord) = patchBorders(1, iCoord) + 2*radius(iCoord);
                if (patchBorders(2, iCoord) > imageDim(iCoord))
                   patchBorders(2, iCoord) = imageDim(iCoord);
                   patchBorders(1, iCoord) = patchBorders(2, iCoord) - 2*radius(iCoord);
                end
           end

           x0 = patchBorders(1, 1); x1 = patchBorders(2, 1);
           y0 = patchBorders(1, 2); y1 = patchBorders(2, 2);
    %        figure(1); 
    %        plot(patchCenters(1, iCenter), patchCenters(2, iCenter), 'r+');
    %        plot([x0 x1 x1 x0 x0], [y0 y0, y1 y1 y0], 'r');

           patchIndex = iCenter + (iScale - 1)*numLocs;
           figure(iPatchType); subplot(numScales, numLocs, patchIndex); imshow(im(y0:y1, x0:x1, :));
           title(num2str(patchIndex));
        end
    end
    
    figure(iPatchType); hold off;
end
% figure(1); hold off;
