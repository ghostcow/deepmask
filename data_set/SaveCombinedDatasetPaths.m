% This script save the dataset image paths into csv file, where each line
% format is : [image path],[label]

clear all variables; clc;
%% change values here
outputFilePath = '../data/deepId/CFW_small2/images';
dirIndices = [2]; % 1 = CFW_clean - CFW_small2, 2 = CFW_small2, 3 = PubFig, 4 = SUFR, 5 = Google

% output txt files
outputFilePathTrain = [outputFilePath '_train.txt'];
outputFilePathTest = [outputFilePath '_test.txt'];
% which part of the images are used for training
trainPerc = 0.8;

%% load all image paths
LoadAllDatasetsPaths;
personNames = nameToLabelMap.keys;
nPersons = length(personNames);

%%
% defining minimum & maximum samples per person
samplesPerPerson = [110 150]; % use [0 inf] to take all images

nPersonsTot = 0;
nImagesTot = 0;
fidTrain = fopen(outputFilePathTrain, 'w');
fidTest = fopen(outputFilePathTest, 'w');
iLabel = 1;

for iPerson = 1:length(personNames)
   personName = personNames{iPerson};
   label = nameToLabelMap(personName);

   fprintf('%d - %s\n', iPerson, personName);
   currPersonImages = imagePaths{label};
   currPersonFaceWidths = faceWidths{label};

    nImages = length(currPersonImages);    
    if (nImages > samplesPerPerson(1))
        if (nImages > samplesPerPerson(2))        
            nImagesToTake = samplesPerPerson(2);
            
            % choose faces with best resolution
            [~, temp] = sort(currPersonFaceWidths, 'descend');
            imageIndices = temp(randperm(nImagesToTake));
        else
            nImagesToTake = nImages;
            imageIndices = randperm(nImages);
        end
        nPersonsTot = nPersonsTot + 1;
        nImagesTot = nImagesTot + nImagesToTake;
        disp(nImagesToTake);
        
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