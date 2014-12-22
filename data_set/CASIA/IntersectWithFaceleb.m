output = 'intersection_FaCeleb_CASIA.txt';
facelebDir = '../FaCeleb';
facelebFileNames = {'names_faceleb_part1.txt', 'names_faceleb_part2.txt', 'names_faceleb_part3.txt'};
casiaFileName = 'names.txt';

% collect all FaCeleb names
facelebNames = {};
for iFile = 1:3
    fid = fopen(fullfile(facelebDir, facelebFileNames{iFile}));
    C = textscan(fid, '%s', 'Delimiter', '_');
    fclose(fid);
    facelebCurrPart = C{1};
    facelebNames = [facelebNames; facelebCurrPart];
end

% collect all CASIA names
casiaNames = {};
fid = fopen(casiaFileName);
C = textscan(fid, '%d %s');
fclose(fid);
casiaNames = C{2};
casiaNames = cellfun(@(x)(strrep(x, '_', ' ')), casiaNames, 'UniformOutput', false);

% subtract FaCeleb from CASIA
x = intersect(facelebNames, casiaNames);
fid = fopen(output, 'w');
for iName = 1:length(x)
    fprintf(fid, '%s\r\n', x{iName});
end
fclose(fid);
