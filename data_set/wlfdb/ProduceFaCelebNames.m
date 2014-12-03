imdbUrl = 'http://www.imdb.com';
peopleFiles = dir('names/people_*.txt');

toLowerFunc = @(x)(lower(strrep(x, ' ','_')));
names = {};
urls = {};
locs = [];
years = [];
for iFile = 12:length(peopleFiles)
    year = str2double(peopleFiles(iFile).name(end-7:end-4));
    fid = fopen(fullfile('names', peopleFiles(iFile).name));
    C = textscan(fid, '%s %s', 'Delimiter', ',');
    fclose(fid);
    
    names = [names; C{1}];
    urls = [urls; C{2}];
    locs = [locs, 1:length(C{1})];
    years = [years, year*ones(1, length(C{1}))];
end
[names, i1, i2] = unique(names);
locs = locs(i1);
years = years(i1);
urls = urls(i1);
processedNames = cellfun(toLowerFunc, names, 'UniformOutput', false);

% remove our dataset people
datasetDirs = {'/media/data/datasets/CFW/images', ...
    '/media/data/datasets/pubfig/images', ...
    '/media/data/datasets/SUFR/images', ...
    '/media/data/datasets/WLFDB/aligned_deepid'};
datasetNames = {};
for iDir = 1:length(datasetDirs)
    dirs = dir(datasetDirs{iDir});
    dirs = dirs(3:end);
    dirs = cellfun(toLowerFunc, {dirs.name}, 'UniformOutput', false);
    if strcmp(datasetDirs{iDir}, '/media/data/datasets/WLFDB/aligned_deepid')
        dirs = cellfun(@(x)(x(6:end)), dirs, 'UniformOutput', false);
    end
    datasetNames = union(datasetNames, dirs);
end
% [processedNames, ia] = setdiff(processedNames, datasetNames);
% check more variations of names
shouldRemove = false(1, length(processedNames));
for iName = 1:length(processedNames)
    if (sum(ismember(datasetNames, processedNames(iName))) > 0) || ...
        (sum(ismember(datasetNames, strrep(processedNames(iName), '_', ''))) > 0) || ...
        (sum(ismember(datasetNames, strrep(processedNames(iName), '''', ''))) > 0)
        shouldRemove(iName) = true;
    end
end
processedNames = processedNames(~shouldRemove);
names = names(~shouldRemove);
locs = locs(~shouldRemove);
years = years(~shouldRemove);
urls = urls(~shouldRemove);

% save final list into file (ordered by locs)
% line format : year[tab]name[tab]url
[locs, indices] = sort(locs);
names = names(indices);
years = years(indices);
urls = urls(indices);

% input file for imdb download script
fid = fopen('names/names_faceleb_1986_1994.txt', 'w');
for iName = 1:length(names)
    fprintf(fid, '%d\t%s\t%s\n', years(iName), names{iName}, [imdbUrl, urls{iName}]);
end
fclose(fid);

% input file for google download script
fid = fopen('../google/names_faceleb_1986_1994.txt', 'w');
for iName = 1:length(names)
    fprintf(fid, '%s\n', names{iName});
end
fclose(fid);