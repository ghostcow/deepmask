alignedImagesDir = '/media/data/datasets/CFW/filtered_aligned';
filteredAlignedImagesDir = '/media/data/datasets/CFW/filtered_aligned_small';
outputFilePath = 'data/images_unknown.txt';
minImagesPerPerson = 42;

%% iterate over the filtered figure dirs
figDirs = dir(alignedImagesDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);
fid = fopen(outputFilePath, 'w');
for iPerson = 1:nPersons
    fprintf('%d - %s\n', iPerson, figDirs(iPerson).name);
    imagesDir = fullfile(alignedImagesDir, figDirs(iPerson).name);
    
    % look for filtered image dir
    filteredImagesDir = fullfile(filteredAlignedImagesDir, figDirs(iPerson).name);
    if exist(filteredImagesDir, 'dir')
        continue;
    end
    
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