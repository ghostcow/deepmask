function [pairIndices, labels] = GetAllPairs(peopleMetadata)

numPairsGuess = 100000;
iPair = 1;

pairIndices = zeros(numPairsGuess, 2);
labels = zeros(1, numPairsGuess);
numPersons = length(peopleMetadata);
for iPerson = 1:numPersons
    fprintf('person %d/%d\n', iPerson, numPersons);
    numImages = peopleMetadata(iPerson).numImages;

    for iImage = 1:numImages 
        % match pairs
        for jImage = (iImage+1):numImages
            pairIndices(iPair, :) = ...
                [peopleMetadata(iPerson).imageIndices(iImage), peopleMetadata(iPerson).imageIndices(jImage)];
            labels(iPair) = 1;
            
            if (mod(iPair, numPairsGuess) == 0)
                pairIndices = [pairIndices; zeros(numPairsGuess, 2)];
                labels = [labels, zeros(1, numPairsGuess)];
            end
            iPair = iPair + 1;
        end
        
        %mismatch pairs
        for jPerson = (iPerson+1):numPersons
            numImages2 = peopleMetadata(jPerson).numImages;
            for jImage = 1:numImages2
                pairIndices (iPair, :) = ...
                    [peopleMetadata(iPerson).imageIndices(iImage), peopleMetadata(jPerson).imageIndices(jImage)];
                labels(iPair) = -1;
                
                if (mod(iPair, numPairsGuess) == 0)
                    pairIndices = [pairIndices; zeros(numPairsGuess, 2)];
                    labels = [labels, zeros(1, numPairsGuess)];
                end
                iPair = iPair + 1;
            end
        end
    end
end
pairIndices(iPair:(size(pairIndices, 1)), :) = [];
labels(iPair:length(labels)) = [];
end