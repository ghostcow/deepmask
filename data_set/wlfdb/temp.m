mainDir = '/media/data/datasets/WLFDB/aligned_deepid';
imagesDir = 'wlfdb_profile_images';

figDirs = dir(mainDir);
figDirs = figDirs(3:end);
for iDir = 1:length(figDirs)
    dirName = figDirs(iDir).name;
    figName = dirName(6:end);
    
    if exist(fullfile(imagesDir, [figName '.jpg']), 'file')
        movefile(fullfile(imagesDir, [figName '.jpg']), ...
            fullfile(imagesDir, [dirName '.jpg']));
    else
        disp(figName);
    end
end