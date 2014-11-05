clc;
% clear all variables;

%% change paths here
mainDirs = {'/media/data/datasets/CFW/filtered_aligned_network/byFigure', '/media/data/datasets/CFW/filtered_aligned_small', ...
    '/media/data/datasets/pubfig/aligned_clean_network_results/byFigure'};
detectionsCfw = ParseDetectionsFile('/media/data/datasets/CFW/detections_CFW.txt');
detectionsPubfig = ParseDetectionsFile('/media/data/datasets/pubfig/detections_pubfig.txt');
isPubFig = [0 0 1];
mapPubFigToCfw = '/media/data/datasets/pubfig/MapToCfwNames.csv';

%% read mapping file between pubfig name to cfw name
fid = fopen(mapPubFigToCfw);
C = textscan(fid, '%s %s', 'delimiter', ',');
fclose(fid);
pubfigNames = C{1}(2:end);
cfwNames = C{2}(2:end);
pubfigToCfwMap = containers.Map;
for iPerson = 1:length(pubfigNames);
    if ~strcmp(cfwNames{iPerson}, '---')
        pubfigToCfwMap(pubfigNames{iPerson}) = cfwNames{iPerson};
    end
end

nPersonsTot = 0;
nImagesTot = 0;
iLabel = 1;
nameToLabelMap = containers.Map;
imagesCount = zeros(1, 1000);
%c('foo') = 1
%c(' not a var name ') = 2
%keys(c)
%values(c)
for iDir = 1:length(mainDirs)
    mainDir = mainDirs{iDir};
    fprintf('%d : %s\n', iDir, mainDir);
    
    % start iterating
    figDirs = dir(mainDir);
    figDirs = figDirs(3:end);
    nPersons = length(figDirs);
    for iPerson = 1:nPersons
        personName = figDirs(iPerson).name;
        isExist = false;
        if isPubFig(iDir)
            isExist = isKey(pubfigToCfwMap, personName) && isKey(nameToLabelMap, pubfigToCfwMap(personName));
        end
        if isExist
            personName = pubfigToCfwMap(personName);
            jLabel = nameToLabelMap(personName);
        else
            nameToLabelMap(personName) = iLabel;
            imagesCount(iLabel) = 0;
            jLabel = iLabel;
            iLabel = iLabel + 1;
        end

        imagesDir = fullfile(mainDir, personName);
        images = dir(fullfile(imagesDir, '*.jpg'));
        nImages = length(images);
        imagesCount(jLabel) = imagesCount(jLabel) + nImages;
        
        % looking for the original image
        for iImage = 1:nImages
            key = fullfile(personName, images(iImage).name);
            if isPubFig(iDir)
                if ~isKey(detectionsPubfig, key)
                    disp(key);
                end                
            else
                % CFW
                if ~isKey(detectionsCfw, key)
                    disp(key);
                end                     
            end

        end
 
    end
end
imagesCount(iLabel:end) = [];

% sum results
disp('Summary :');
personNames = nameToLabelMap.keys;
nPersons = length(personNames);
labelToNameMap = cell(1, nPersons);
for iPerson = 1:length(personNames)
   personName = personNames{iPerson};
   nImages = imagesCount(nameToLabelMap(personName));
   labelToNameMap{nameToLabelMap(personName)} = personName;
   fprintf('%s,%d\n', personName, nImages);
end