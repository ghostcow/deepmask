namesFiles = {'names_faceleb_1975_1985.txt', 'names_faceleb_1986_1994.txt'};
outputFile = 'names/names_faceleb_no_LFW.txt';
toLowerFunc = @(x)(lower(strrep(x, ' ','_')));

names = {};
locs = [];
for iFile = 1:2
    fid = fopen(fullfile('names', namesFiles{iFile}));
    line = fgetl(fid);
    iLoc = 1;
    while ischar(line)
        names{end+1} = line;
        locs(end+1) = iLoc;
        iLoc = iLoc + 1;
        line = fgetl(fid);
    end
end
processedNames = cellfun(toLowerFunc, names, 'UniformOutput', false);

% Get LFW names
lfwDirs = dir('/media/data/datasets/LFW/lfw');
lfwDirs = lfwDirs(3:end);
lfwNamesProcessed = cellfun(toLowerFunc, {lfwDirs.name}, 'UniformOutput', false);

% filter out LFW
[~, indices] = setdiff(processedNames, lfwNamesProcessed);
names = names(indices);
locs = locs(indices);
% sort by location in the original text file
[~, sortedIndices] = sort(locs);
names = names(sortedIndices);
% put highr the images with no images produced from imdb
nImagesPerName = zeros(1, length(names));
imagesDir = '/media/data/datasets/FaCeleb_75_85/images';
dirs = dir(imagesDir);
dirs = dirs(3:end);
for iDir = 1:length(dirs)
    name = dirs(iDir).name(7:end);
    k = ismember(names, name);
    nImagesPerName(k) = length(dir(fullfile(imagesDir, dirs(iDir).name, '*.jpg')));
end
[~, sortedIndices] = sort(nImagesPerName);
names = names(sortedIndices);

% write output
fid = fopen(outputFile, 'w');
for iName = 1:length(names)
    fprintf(fid, [names{iName} '\r\n']);
end
fclose(fid);