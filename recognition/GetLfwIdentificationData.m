function [galleryX, galleryY, probeX, probeY] = ...
    GetLfwIdentificationData(lfwpeopleResFilePath, lfwPeopleImagesFilePath, netIndices)

if ~exist('netIndices', 'var')
    netIndices = [];
end

% Load LFW people (metadata & features) - 
[X, Y, ~, imPaths] = LoadLfwPeople(lfwpeopleResFilePath, lfwPeopleImagesFilePath, netIndices);
    % build a map between image name to index in X/Y
mapNamesToIndex = containers.Map; 
for iImage = 1:length(Y)
    [~, imageName, ext] = fileparts(imPaths{iImage});
    mapNamesToIndex([imageName, ext]) = iImage;
end
% Load protocol text file
currFunDir = fileparts(mfilename('fullpath'));
fid = fopen(fullfile(currFunDir, 'lfw_protocols', 'closedset_gallery_4249.txt'));
C = textscan(fid, '%s');
fclose(fid);
namesGallery = C{1};

fid = fopen(fullfile(currFunDir, 'lfw_protocols', 'closedset_probe_3143.txt'));
C = textscan(fid, '%s');
fclose(fid);
namesProbe = C{1};

%
galleryX = zeros(size(X, 1), length(namesGallery));
galleryY = zeros(1, length(namesGallery));
probeX = zeros(size(X, 1), length(namesProbe));
probeY = zeros(1, length(namesProbe));

for iSet = 1:2
    switch iSet
        case 1 
            imageNames = namesGallery;
        case 2
            imageNames = namesProbe;
    end
    peopleNames = cellfun(@(x)(x((1:end-9))), imageNames, 'UniformOutput', false);
    
    if (iSet == 2)
        % remove from probe all faces with invalid face in gallery
        invalidImages = ismember(peopleNames, invalidPeopleNames);
        imageNames = imageNames(~invalidImages);
        peopleNames = peopleNames(~invalidImages);
    end
    
    x = zeros(size(X, 1), length(imageNames));
    y = zeros(1, length(imageNames));
    for iName = 1:length(imageNames)
        % invalid images (failed face detection) doesn't exist in mapNamesToIndex
        if isKey(mapNamesToIndex, imageNames{iName})
            imageIndex = mapNamesToIndex(imageNames{iName});
            x(:, iName) = X(:, imageIndex);
            y(iName) = Y(imageIndex);
        end
    end
    
    if (iSet == 1)
        % save invalid people names, so they will be filtered out from the
        % probe later
        invalidPeopleNames = peopleNames(y == 0); 
    end
    
    % remove from set invalid faces
    validIndices = find(y > 0);
    x = x(:, validIndices);
    y = y(:, validIndices);
        
    switch iSet
        case 1 
            galleryX = x;
            galleryY = y;
        case 2
            probeX = x;
            probeY = y;
    end
end