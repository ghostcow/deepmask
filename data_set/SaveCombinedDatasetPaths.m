% This script save the dataset image paths into csv file, where each line
% format is : [image path],[label]

clear variables; 
clc;

%% change values here
name = 'casia';
% defining minimum & maximum samples per person (relevant only for the nn training)
samplesPerPerson = [0 inf]; % take all images
verificationPerc = 0; % how many persons will be used for learning the verification model

%%
% which part of the images are used for training
trainPerc = 0.9;

if strcmp(name, 'cfw_small2')
    outputFilePath = '../data/deepId/CFW_small2/images';
    dirIndices = 2; 
    samplesPerPerson = [110 150]; 
elseif strcmp(name, 'cfw_pubfig')
    outputFilePath = '../data/deepId/CFW_PubFig/images';
    dirIndices = [1:3, 5]; 
    samplesPerPerson = [18 40];
elseif strcmp(name, 'cfw_pubfig_sufr')
    type = 'deepid'; % different size of images
    outputFilePath = '../data/deepId_full/CFW_PubFig_SUFR/images';
    dirIndices = 1:5; 
    samplesPerPerson = [14 40]; 
    verificationPerc = 0.2;
elseif strcmp(name, 'casia')
    type = 'deepid'; % different size of images
    outputFilePath = '../data/deepId_full/CASIA/images';
    dirIndices = 8; 
    samplesPerPerson = [0 inf]; 
    verificationPerc = 0;
end

mkdir(fileparts(outputFilePath));
% output txt files
outputFilePathTrain = [outputFilePath '_train.txt'];
outputFilePathTest = [outputFilePath '_test.txt'];

%% load all image paths
LoadAllDatasetsPaths;
personNames = nameToLabelMap.keys;

% remove overlapping persons with LFW
fid = fopen('../data/LFW/people.txt');
C = textscan(fid, '%s %d');
fclose(fid);
lfwNames = C{1}(3:end);
lfwNames = cellfun(@lower, lfwNames, 'UniformOutput', false);

% here we could use intersect(personNames, lfwNames), but we will use the folder names to get more variations
overlapPersons = {}; 
for iPerson = 1:length(personNames)
    personName = personNames{iPerson};
    label = nameToLabelMap(personName);
    currPersonImages = imagePaths{label};
    dirNames = cell(1, length(currPersonImages));
    for iImage = 1:length(currPersonImages)
        imPath = currPersonImages{iImage};
        temp = fileparts(imPath);
        [~, figName] = fileparts(temp);
        dirNames{iImage} = lower(figName);
    end
    dirNames = unique(dirNames);
    if ~isempty(intersect(dirNames, lfwNames))
        overlapPersons{end+1} = personName;
        continue;
    end
end
personNames = setdiff(personNames, overlapPersons);
nPersons = length(personNames);

%% split to nn/verification training
if (verificationPerc > 0)
    % how many people will be used for the neural network training ?
    nPersonsNn = round(nPersons*(1-verificationPerc));
    % how many people will be used for learning the verification model ?
    nPersonsVer = nPersons - nPersonsNn;
    
    personsIndices = randperm(nPersons);
    personNamesNn = personNames(personsIndices(1:nPersonsNn));
    personNamesVer = personNames(personsIndices((nPersonsNn+1):end));
else
   personNamesNn = personNames;
end

%% data for nn training
fprintf('label,name,#images\n');
fprintf('\ndata for neural network (train + test)\n');
fidTrain = fopen(outputFilePathTrain, 'w');
fidTest = fopen(outputFilePathTest, 'w');
iLabel = 1;
for iPerson = 1:length(personNamesNn)
   personName = personNamesNn{iPerson};
   label = nameToLabelMap(personName);

   currPersonImages = imagePaths{label};
   currPersonFaceWidths = faceWidths{label};

    nImages = length(currPersonImages);    
    if (nImages >= samplesPerPerson(1)) 
        if (nImages > samplesPerPerson(2))        
            nImagesToTake = samplesPerPerson(2);
            
            % choose faces with best resolution
            [~, temp] = sort(currPersonFaceWidths, 'descend');
            imageIndices = temp(randperm(nImagesToTake));
        else
            nImagesToTake = nImages;
            imageIndices = randperm(nImages);
        end
        fprintf('%d,%s,%d\n', iLabel, personName, nImagesToTake);
        
        nTrainingSamples = round(nImagesToTake*trainPerc);
        % write image paths to files
        for iImage = 1:nImagesToTake
            imagePath = currPersonImages{iImage};
            if (iImage <= nTrainingSamples)
                fprintf(fidTrain, '%s,%d\n', imagePath, iLabel);
            else
                fprintf(fidTest, '%s,%d\n', imagePath, iLabel);
            end
        end
         iLabel = iLabel + 1;
    end  
end
fclose(fidTrain);
fclose(fidTest);

%% data for verification model training
if (verificationPerc > 0)
    fprintf('\ndata for face verification model\n');
    outputFilePathVer = [outputFilePath '_verification.txt'];
    fidVer = fopen(outputFilePathVer, 'w');
    for iPerson = 1:length(personNamesVer)
        personName = personNamesVer{iPerson};
        label = nameToLabelMap(personName);
        currPersonImages = imagePaths{label};
        
        % all images are taken for the verification model, without upsampling/downsampling
        nImages = length(currPersonImages);   
        
        fprintf('%d,%s,%d\n', iLabel, personName, nImages);
        for iImage = 1:nImages
            imagePath = currPersonImages{iImage};
            fprintf(fidVer, '%s,%d\n', imagePath, iLabel);
        end
         iLabel = iLabel + 1;
    end
    fclose(fidVer);
end