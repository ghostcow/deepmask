resultFilePath = './data/images_unknown_results_NLL.txt';
targetClass2Dir = '/media/data/datasets/CFW/filtered_aligned_network/background';

fid = fopen(resultFilePath);
C = textscan(fid, '%s %f %f', 'delimiter', ',');
fclose(fid);

imagePaths = C{1};
nImages = length(imagePaths);
class2Probs = C{3};

% sort by desccending background score
[sortedScores, imageIndices] = sort(class2Probs, 'descend');
numImagesPerDir = 1000;
numDirs = ceil(nImages / numImagesPerDir);
for iDir = 1:numDirs
   iStart = 1 + (iDir-1)*numImagesPerDir;
   iEnd = min(iStart + numImagesPerDir - 1, nImages);
   
   dirName = ['set' num2str(iDir) '_probs_' ...
       num2str(sortedScores(iStart)) '_' num2str(sortedScores(iEnd))];
   mkdir(fullfile(targetClass2Dir, dirName));
   
   dirImageIndices = imageIndices(iStart:iEnd);
   for k = 1:length(dirImageIndices);
       iImage = dirImageIndices(k);
       imagePath =  imagePaths{iImage};
       [imageDir, imageName, imageExt] = fileparts(imagePath);
       [~, figureName] = fileparts(imageDir);
       
       copyfile(imagePath, fullfile(targetClass2Dir, dirName, [figureName '.' imageName, imageExt]));
   end
end
