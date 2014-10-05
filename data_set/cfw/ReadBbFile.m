function gtFaces = ReadBbFile(bbFilePath)

gtFaces = containers.Map;

fid = fopen(bbFilePath);
while ~feof(fid)
    line = fgetl(fid); %# read line by line
    C = strsplit(line);
    
    if (length(C) >= 5)
        % at least one face
        nFaces = (length(C) - 1) / 4;
        faces = struct('center', {}, 'width', {}, 'height', {});
        for iFace = 1:nFaces
            faceIndex = 2 + (iFace-1)*4;
            bbox = [str2double(C{faceIndex}), str2double(C{faceIndex+1}), ...
                str2double(C{faceIndex+2}), str2double(C{faceIndex+3})];
            faces(iFace).center = [bbox(1) + bbox(2), bbox(3) + bbox(4)] / 2;
            faces(iFace).width = bbox(2) - bbox(1);
            faces(iFace).height = bbox(4) - bbox(3);
        end
        
        imageName = C{1};
        gtFaces(imageName) = faces;
    end
end

fclose(fid);

end