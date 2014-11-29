addpath(genpath('../img_preproc'));

%% change paths here
if ~exist('type', 'var')
    type = 'deepface';
end

if strcmp(type, 'deepface')
    % 1 - CFW, 2 - CFW_small2, 3 - PubFig, 4 - SUFR, 5 - missing_cfw_pubfig (google), 6 - missing_sufr (google)
    mainDirs = {'/media/data/datasets/CFW/filtered_aligned_network/byFigure', ...
        '/media/data/datasets/CFW/filtered_aligned_small', ...
        '/media/data/datasets/pubfig/aligned_clean_network_results/byFigure', ...
        '/media/data/datasets/SUFR/byFigure', ...
        '/media/data/datasets/missing_pubfig_cfw/aligned', ...
        '/media/data/datasets/missing_sufr/aligned', ...
        '/media/data/datasets/LFW/lfw_aligned'};
    detectionsFilePaths = {'/media/data/datasets/CFW/detections_CFW.txt', ...
        '/media/data/datasets/CFW/detections_CFW.txt', ...
        '/media/data/datasets/pubfig/detections_pubfig.txt', ...
        '/media/data/datasets/SUFR/detections_SUFR.txt', ...
        '/media/data/datasets/missing_pubfig_cfw/detections_missing_pubfig_cfw.txt', ...
        '/media/data/datasets/missing_sufr/detections_missing_sufr.txt', ...
        '/media/data/datasets/LFW/detections_lfw.txt'};
elseif strcmp(type, 'deepid')
    % different size of images
     mainDirs = {'/media/data/datasets/CFW/aligned_deepid', ...
        '/media/data/datasets/pubfig/aligned_deepid', ...
        '/media/data/datasets/SUFR/aligned_deepid', ...
        '/media/data/datasets/missing_pubfig_cfw/aligned_deepid', ...
        '/media/data/datasets/missing_sufr/aligned_deepid'};
    detectionsFilePaths = {'/media/data/datasets/CFW/detections_CFW.txt', ...
        '/media/data/datasets/pubfig/detections_pubfig.txt', ...
        '/media/data/datasets/SUFR/detections_SUFR.txt', ...
        '/media/data/datasets/missing_pubfig_cfw/detections_missing_pubfig_cfw.txt', ...
        '/media/data/datasets/missing_sufr/detections_missing_sufr.txt'};   
end

if ~exist('dirIndices', 'var')
    dirIndices = 1:length(mainDirs); % which dirs to use
end
mainDirs = mainDirs(dirIndices);
detectionsFilePaths = detectionsFilePaths(dirIndices);

%% Load metadata
detections = cell(1, length(mainDirs));
mapNames = cell(1, length(mainDirs));
for iDir = 1:length(mainDirs)
    detectionsTxtFilePath = detectionsFilePaths{iDir};
    detectionsMatFilePath = [detectionsTxtFilePath(1:end-3) 'mat'];
    if exist(detectionsMatFilePath, 'file')
        % load save mat file
        S = load(detectionsMatFilePath);
        detections{iDir} = S.detections_;
    else
        detections_ = ParseDetectionsFile(detectionsTxtFilePath);
        save(detectionsMatFilePath, 'detections_');
        detections{iDir} = detections_;
    end
    
    mapNamesFilePath = fullfile(fileparts(detectionsTxtFilePath), 'MapToCfwNames.csv');
    if (exist(mapNamesFilePath, 'file')) % mapping names file exists
        mapNames{iDir} = ParseMapNamesFile(mapNamesFilePath);
    end
end

%% read mapping file between pubfig name to cfw name
nPersonsTot = 0;
nImagesTot = 0;
iLabel = 1;
nameToLabelMap = containers.Map;
imagePaths = cell(1, 1500);
imagesCount = zeros(1, 1500);
faceWidths = cell(1, 1500);
for iDir = 1:length(mainDirs)
    mainDir = mainDirs{iDir};
    fprintf('%d : %s\n', iDir, mainDir);
        
    % start iterating
    figDirs = dir(mainDir);
    figDirs = figDirs(3:end);
    nPersons = length(figDirs);
    for iPerson = 1:nPersons
        personName = figDirs(iPerson).name;
        if (~isempty(mapNames{iDir})) % names mapping file exists
            if isKey(mapNames{iDir}, personName)
                personName = mapNames{iDir}(personName);
            end
        end
        
        isExist = isKey(nameToLabelMap, personName);
        if isExist
            jLabel = nameToLabelMap(personName);
        else
            nameToLabelMap(personName) = iLabel;
            imagePaths{iLabel} = {};
            faceWidths{iLabel} = [];
            imagesCount(iLabel) = 0;
            jLabel = iLabel;
            
            iLabel = iLabel + 1;
        end

        imagesDir = fullfile(mainDir, figDirs(iPerson).name);
        images = dir(fullfile(imagesDir, '*.jpg'));
        nImages = length(images);
        % looking for the original image
        % sort paths by face resolution
        for iImage = 1:nImages
            key = fullfile(figDirs(iPerson).name, images(iImage).name);
            if ~isKey(detections{iDir}, key)
                disp(key);
            else
                currImageDetection = detections{iDir}(key);
                
                imagePaths{jLabel}{imagesCount(jLabel) + iImage} = ...
                    fullfile(mainDir, figDirs(iPerson).name, images(iImage).name);
                faceWidths{jLabel}(imagesCount(jLabel) + iImage) = ...
                    currImageDetection.detection(3); % half-width of the face detected      
            end
        end
        imagesCount(jLabel) = imagesCount(jLabel) + nImages;
    end
end
imagesCount(iLabel:end) = [];
imagePaths(iLabel:end) = [];
faceWidths(iLabel:end) = [];

% build map object between label and name
personNames = nameToLabelMap.keys;
nPersons = length(personNames);
labelToNameMap = cell(1, nPersons);
for iPerson = 1:length(personNames)
   personName = personNames{iPerson};
   labelToNameMap{nameToLabelMap(personName)} = personName;
end

minImages = 15;
fprintf('peoples with less than %d images:\n', minImages);
k = find(imagesCount < minImages);
for iLabel = k
    disp(labelToNameMap{iLabel});
end