filteredAlignedImagesDir = '/media/data/datasets/CFW/filtered_aligned_small';
targetDir = '/media/data/datasets/CFW/clean_aligned_faces_network/true';

% iterate over the filtered figure dirs
figDirs = dir(filteredAlignedImagesDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);

for iPerson = 1:nPersons
    fprintf('%d - %s\n', iPerson, figDirs(iPerson).name);
    
    % dir all images (not filtered)
    imagesDir = fullfile(filteredAlignedImagesDir, figDirs(iPerson).name);
    images = dir(fullfile(imagesDir, '*.jpg'));
    nImages = length(images);

    for iImage = 1:nImages
        imageName = images(iImage).name;
        imagePath = fullfile(imagesDir, imageName);
        
        % check for the same image in the filtered images
        targetPath = fullfile(targetDir, [figDirs(iPerson).name '.' imageName]);
        copyfile(imagePath, targetPath);
    end

end  