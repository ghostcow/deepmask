% Load lfw & our dataset names
existNames = {};
mainDir = '/media/data/datasets';
figDirs = dir(fullfile(mainDir, 'LFW/lfw_aligned_deepid'));
figDirs = figDirs(3:end);
currNames = {figDirs.name};
currNames = cellfun(@(x)(lower(strrep(x, ' ', '_'))), currNames, 'UniformOutput', false);
existNames = [existNames, currNames];

dataPaths = {'../../data/deepId/CFW_PubFig_SUFR/images_train.txt', ...
    '../../data/deepId/CFW_PubFig_SUFR/images_verification.txt'};
datasetNames = {};
for iFile = 1:2
    fid = fopen(dataPaths{iFile});
    C = textscan(fid, '%s %d', 'Delimiter', ',');
    fclose(fid);
    for iName = 1:length(C{1})
        path = C{1}{iName};
        temp = fileparts(path);
        [~, name] = fileparts(temp);
        datasetNames{end+1} = lower(strrep(name, ' ', '_'));
    end
end
existNames = [existNames, unique(datasetNames)];

% 
% dataDirs = {'CFW/aligned_deepid', 'pubfig/aligned_deepid', 'SUFR/aligned_deepid', ...
%     'WLFDB/aligned_deepid', 'LFW/lfw_aligned_deepid'};
% for iDir = 1:length(dataDirs)
%     figDirs = dir(fullfile(mainDir, dataDirs{iDir}));
%     figDirs = figDirs(3:end);
%     currNames = {figDirs.name};
%     currNames = cellfun(@(x)(lower(strrep(x, ' ', '_'))), currNames, 'UniformOutput', false);
%     existNames = [existNames, currNames];
% end

% name	image_id	face_id	url	bbox	sha256
inputFiles = {'facescrub_actors.txt', 'facescrub_actresses.txt'};
fid2 = fopen('facescrub_unique.txt', 'w');
names = {};
for iFile = 1:2
    inputFile = inputFiles{iFile};
    fid = fopen(inputFile);
    line = fgetl(fid);
    line = fgetl(fid);
    while ischar(line)
        C = textscan(line, '%s %d %d %s %s %s', 'Delimiter', '\t');
        if length(C{1}) > 1
            disp(line);
        end
        name = C{1}{1};
        % mkdir(fullfile('/media/data/datasets/FaceScrub/aligned_deepid', name));
        name = lower(strrep(name, ' ', '_'));
        if ~ismember(existNames, name)
            fprintf(fid2, [line '\r\n']);
        end
        names{end+1} = name;
        line = fgetl(fid);
    end
    fclose(fid);
end
names = unique(names);
fclose(fid2);
X = setdiff(names, existNames);