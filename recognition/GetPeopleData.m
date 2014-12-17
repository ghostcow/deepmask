function peopleMetadata = GetPeopleData()
    currFunDir = fileparts(mfilename('fullpath'));
    filePath = fullfile(currFunDir, 'lfw_protocols', 'people.txt');
    
    peopleMetadata = {};
    
    numFolds = -1;
    currFoldSize = -1;
    iFoldPerson = 0;
    
    iImage = 1;
    fid = fopen(filePath);
    line = fgetl(fid);
    while ischar(line)
        if (numFolds == -1)
            % first line
            numFolds = str2double(line);
        elseif (currFoldSize == -1) 
            % new fold line
            currFoldSize = str2double(line);
            currFold = struct('numImages', cell(1, currFoldSize), 'imageIndices', cell(1, currFoldSize));
        else
            % during fold
            k = regexp(line, '\t');
            personName = line(1:(k-1));
            numImages = str2double(line((k+1):end));
            iFoldPerson = iFoldPerson + 1;
            currFold(iFoldPerson).name = personName;
            currFold(iFoldPerson).numImages = numImages;
            currFold(iFoldPerson).imageIndices = iImage:(iImage + numImages - 1);
            iImage = iImage + numImages;
            
            if (iFoldPerson == currFoldSize)
                % starting new fold
                currFoldSize = -1;
                iFoldPerson = 0;
                peopleMetadata{end+1} = currFold;
            end            
        end
        
        line = fgetl(fid);
    end
    fclose(fid);


end