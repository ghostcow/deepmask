mainDir = '/media/data/datasets/CFW/filtered_aligned_network/byFigure';
additionalImagesDir = '/media/data/datasets/CFW/filtered_aligned_network/background';
outputDir = '/media/data/datasets/CFW/filtered_aligned_network/byFigureMore';

figDirs = dir(mainDir);
figDirs = figDirs(3:end);
nPersons = length(figDirs);
nSamplesPerPerson = zeros(1, nPersons);

additionalImagesDirs = dir(additionalImagesDir);
additionalImagesDirs = additionalImagesDirs(3:end);

for iPerson = 1:nPersons
    imagesDir = fullfile(mainDir, figDirs(iPerson).name);
    images = dir(fullfile(imagesDir, '*.jpg'));
    
    nImages = length(images);
    nSamplesPerPerson(iPerson) = nImages;
    if (nImages < 30)
        fprintf('%s - %d\n', figDirs(iPerson).name, nImages);
        
        %% look for additional images
        for iDir = 1:numel(additionalImagesDirs)
            imagesPer = dir(fullfile(additionalImagesDir, additionalImagesDirs(iDir).name, [figDirs(iPerson).name '*']));
            for iImage = 1:numel(imagesPer)
                movefile(fullfile(additionalImagesDir, additionalImagesDirs(iDir).name, imagesPer(iImage).name), ...
                    fullfile(outputDir, imagesPer(iImage).name));
            end
        end
    end
end