mainDir = '/media/data/datasets/';
pubfigDir = fullfile(mainDir, 'pubfig', 'images');
cfwDir = fullfile(mainDir, 'CFW', 'images');
outputFilePath = fullfile(mainDir, 'pubfig', 'MapToCfwNames.csv');

pubfigDirs = dir(pubfigDir);
pubfigDirs = pubfigDirs(3:end);
fid = fopen(outputFilePath, 'w');
fprintf(fid, 'PubFig_name,CFW_name\n');
for iPubFigName = 1:numel(pubfigDirs)
    pubfigName = pubfigDirs(iPubFigName).name;
    
    if exist(fullfile(cfwDir, pubfigName), 'dir')
        cfwName = pubfigName;
    elseif exist(fullfile(cfwDir, lower(pubfigName)), 'dir')
        cfwName = lower(pubfigName);
    else
        cfwName = '---';
    end
    fprintf(fid, '%s,%s\n', pubfigName, cfwName);
end
fclose(fid);