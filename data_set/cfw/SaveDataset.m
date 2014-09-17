mainDir = '/media/data/datasets/CFW/filtered_faces';
figDirs = dir(mainDir);
figDirs = figDirs(3:end);
imSize = [152,152];

%% small dataset - use only 200 persons
nPersons = 200;
trainPerc = 0.7;

train = zeros(0, 3, imSize(1), imSize(2));
trainLabels = [];
test = zeros(0, 3, imSize(1), imSize(2));
testLabels = [];

for iFigure = 1:nPersons
    disp(iFigure);
    currDir = fullfile(mainDir, figDirs(iFigure).name);
    images = dir(fullfile(currDir, '*.png'));
    nImages = length(images);
    nTraining = round(nImages*trainPerc);
    isTrain = [ones(nTraining, 1); zeros(nImages - nTraining, 1)];
    
    currFigTrain = zeros(0, 3, imSize(1), imSize(2));
    currFigTest = zeros(0, 3, imSize(1), imSize(2));
    
    for iImage = 1:nImages
        [im, map] = imread(fullfile(currDir, images(iImage).name));
        if (size(im, 3) == 1)
            if ~isempty(map)
                im = ind2rgb(im, map);
            else
                error('grayscale image');
            end
        end
        
        im = im2double(im);
        im = imresize(im, imSize);
        % convert shape from 152x152x3 3x152x152
        % pay attention that this operation also flip between height&width
        im = shiftdim(im, 2); 
        
        if isTrain(iImage)
            currFigTrain(end+1,:,:,:) = im;
        else
            currFigTest(end+1,:,:,:) = im;
        end
    end
    
    % copy into global arrays
    train = cat(1, train, currFigTrain);
    trainLabels = cat(1, trainLabels, iFigure*ones(size(currFigTrain,1), 1));
    test = cat(1, test, currFigTest);
    testLabels = cat(1, testLabels, iFigure*ones(size(currFigTest,1), 1));
end
save('cfw_small.mat', 'train', 'trainLabels', 'test', 'testLabels');