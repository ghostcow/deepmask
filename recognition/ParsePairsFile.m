function [namesLeft, namesRight, imgNumLeft, imgNumRight] = ...
    ParsePairsFile(pairsFilePath)

if ~exist('pairsFilePath', 'var')
    pairsFilePath = '/media/data/datasets/LFW/view2/pairs.txt';
end

numFolds = -1;
numPairsPerFold = -1;
iPair = 0;

fid = fopen(pairsFilePath);
line = fgetl(fid);
while ischar(line)
    if (numFolds == -1)
        % first line
        C = textscan(line, '%d %d');
        numFolds = C{1};
        numPairsPerFold = C{2};
        
        numPairs = numFolds*numPairsPerFold;
        namesLeft = cell(1, numPairs);
        namesRight = namesLeft;
        imgNumLeft = zeros(1, numPairs);
        imgNumRight = imgNumLeft;
    else
        % during fold
        if (mod(iPair, 2*numPairsPerFold) < numPairsPerFold)
            % positive pair
            C = textscan(line, '%s %d %d');
            namesLeft{iPair+1} = C{1}{1};
            namesRight{iPair+1} = namesLeft{iPair+1};
            imgNumLeft(iPair+1) = C{2};
            imgNumRight(iPair+1) = C{3};
        else
            C = textscan(line, '%s %d %s %d');
            namesLeft{iPair+1} = C{1}{1};
            namesRight{iPair+1} = C{3}{1};
            imgNumLeft(iPair+1) = C{2};
            imgNumRight(iPair+1) = C{4};            
        end
        iPair = iPair + 1;
    end
    line = fgetl(fid);
end
fclose(fid);

end