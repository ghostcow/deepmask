name = 'wlfdb';

landmarksFiles = dir(['landmarks_' name '_*.mat']);
poseFiles = dir(['pose_' name '_*.txt']);

combinedLandmarksFile = ['landmarks_' name '.mat'];
poseFile = ['pose_' name '.txt'];
mapFile = 'pose_results.mat';

% poseResults = containers.Map;
% isKey(poseResults, key)

% combine all pose files
if false
fido = fopen(poseFile, 'w');
for iFile = 1:length(poseFiles)
    fid = fopen(poseFiles(iFile).name);
    line = fgetl(fid);
    while ischar(line)
        fprintf(fido, [line '\r\n']);
        C = textscan(line, '%s %d', 'Delimiter', ',');
        path = C{1}{1};
        pose = C{2};
        
        [temp, fileName, ext1] = fileparts(path);
        [~, dirName, ext2] = fileparts(temp);
        key = fullfile([dirName ext2], [fileName ext1]);
        poseResults(key) = struct('pose', pose);
        line = fgetl(fid);
    end
    fclose(fid);
end
fclose(fido);
end

% combine all landmarks files
landmarks = [];
for iFile = 1:length(landmarksFiles)
    S = load(landmarksFiles(iFile).name);
    
end