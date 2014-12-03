detections_ = containers.Map; 
inputFiles = {'facescrub_actors.txt', 'facescrub_actresses.txt'};
for iFile = 1:2
    fid = fopen(inputFiles{iFile});
    line = fgetl(fid);
    line = fgetl(fid);
    while ischar(line)
        C = textscan(line, '%s %d %d %s %s %s', 'Delimiter', '\t');
        name = C{1}{1};
        id = C{2};
        bboxStr = C{5}{1};
        bbox = textscan(bboxStr, '%d %d %d %d', 'Delimiter', ',');
        bbox = cell2mat(bbox);
        bbox = bbox + 1;

        % bbox = [x1,y1,x2,y2]
        y1 = bbox(2); y2 = bbox(4);
        x1 = bbox(1); x2 = bbox(3);
        width = x2 - x1 + 1;
        height = y2 - y1 + 1;

        key = fullfile(name, sprintf('%06d.jpg', id));
        detection = [x1, y1, width/2, 1];
        detections_(key) = detection;
        line = fgetl(fid);
    end
    fclose(fid);
end
save('/media/data/datasets/FaceScrub/detections_facescrub.mat', 'detections_');
