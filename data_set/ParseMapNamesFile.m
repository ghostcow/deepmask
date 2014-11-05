function mapNames = ParseMapNamesFile(filePath)

fid = fopen(filePath);
C = textscan(fid, '%s %s', 'delimiter', ',');
fclose(fid);

names = C{1}(2:end);
mappedNames = C{2}(2:end);
mapNames = containers.Map;
for iPerson = 1:length(names);
    if ~strcmp(mappedNames{iPerson}, '---')
        if strcmp(mappedNames{iPerson}, '=')
            % same name
            mapNames(names{iPerson}) = names{iPerson};
        else
            mapNames(names{iPerson}) = mappedNames{iPerson};
        end
    end
end
