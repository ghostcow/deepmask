alignedImagesDir = '/media/data/datasets/pubfig/aligned';
outputFilePath = 'data/images_unknown_pubfig.txt';
minImagesPerPerson = 5;

%% iterate over the filtered figure dirs
figDirs = dir(alignedImagesDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);
fid = fopen(outputFilePath, 'w');
for iPerson = 1:nPersons
    fprintf('%d - %s\n', iPerson, figDirs(iPerson).name);
    imagesDir = fullfile(alignedImagesDir, figDirs(iPerson).name);
    
    % dir all images (not filtered)
    images = dir(fullfile(imagesDir, '*.jpg'));
    nImages = length(images);
    if (nImages < minImagesPerPerson)
        continue;
    end
    for iImage = 1:nImages
        imageName = images(iImage).name;
        imagePath = fullfile(imagesDir, imageName);
        fprintf(fid, '%s\n', imagePath);
    end

end  
fclose(fid);