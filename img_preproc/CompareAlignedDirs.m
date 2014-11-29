clc;
close all

mainDirs1 = {'/media/data/datasets/CFW/filtered_aligned_network/byFigure', ...
    '/media/data/datasets/CFW/filtered_aligned_small', ...
    '/media/data/datasets/pubfig/aligned_clean_network_results/byFigure', ...
    '/media/data/datasets/SUFR/byFigure', ...
    '/media/data/datasets/missing_pubfig_cfw/aligned', ...
    '/media/data/datasets/missing_sufr/aligned'};

mainDirs2 = {'/media/data/datasets/CFW/aligned_deepid', ...
    '/media/data/datasets/CFW/aligned_deepid', ...
    '/media/data/datasets/pubfig/aligned_deepid', ...
    '/media/data/datasets/SUFR/aligned_deepid', ...
    '/media/data/datasets/missing_pubfig_cfw/aligned_deepid', ...
    '/media/data/datasets/missing_sufr/aligned_deepid'};

for iDir = 3:length(mainDirs1)
    mainDir1 = mainDirs1{iDir};
    fprintf('%s\n', mainDir1);
    
    mainDir2 = mainDirs2{iDir};
	figDirs = dir(mainDir1);
    figDirs = figDirs(3:end);
    nPersons = length(figDirs);
    for iPerson = 1:nPersons
        fprintf('%d : %s\n', iPerson, figDirs(iPerson).name);
        imagesDir = fullfile(mainDir1, figDirs(iPerson).name);
        images = dir(fullfile(imagesDir, '*.jpg'));
        
        imagesDir2 = fullfile(mainDir2, figDirs(iPerson).name);
        images2 = dir(fullfile(imagesDir, '*.jpg'));
        for iImage = 1:length(images)
            imPath1 = fullfile(imagesDir, images(iImage).name);
            imPath2 = fullfile(imagesDir2, images(iImage).name);
            if ~exist(imPath2, 'file');
                disp(imPath1);
            end
        end
    end
end