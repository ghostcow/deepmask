function [resDir, lfwpairsResFileName, lfwpeopleResFileName, verificationResFileName, verificationImagesFilePath] = ...
     GetResultFilePaths(resIndex)

% resIndex : 
%   1 - first training of 5 deepID networks, the input are patched cropped
%       from the deepface aligned images
%   2 - same as 1, after adding missing RELU in the representation layer
%   PAY ATTENTION - resIndex 1&2 should be ignored because some LFW identities were used for training
%   
%   3 - training over 15 deepID networks, the input are patcehs cropped
%       from the deepid aligned images.
%       The training data is CFW+PubFig+SUFR, after subtracting some LFW
%       identities.

% deepface type patches (152x152)
switch resIndex
    case 1
        % PAY ATTENTION : bad results, becuase some LFW identities were
        % used for training
        resDir = '../results_deepid/CFW_PubFig_SUFR_deepID.3.64_dropout_flipped';
        lfwpairsResFileName = 'deepid_LFW_pairs_patch*';
        lfwpeopleResFileName = 'deepid_LFW_people_patch*';
        verificationResFileName = 'deepid_CPS_verification_patch*';
        verificationImagesFilePath = '../data/deepId/CFW_PubFig_SUFR/images_verification.txt';
    case 2
        % PAY ATTENTION : bad results, becuase some LFW identities were
        % used for training        
        resDir = '../results_deepid/CFW_PubFig_SUFR_deepID.3.64_dropout_flipped_ReLu';
        lfwpairsResFileName = 'LFW_pairs_patch*';
        lfwpeopleResFileName = 'LFW_people_patch*';
        verificationResFileName = 'verification_patch*';
        verificationImagesFilePath = '../data/deepId/CFW_PubFig_SUFR/images_verification.txt';    
    case 3
        resDir = '../results_deepid/CFW_PubFig_SUFR_deepID.3.64_15patches/features';
        lfwpairsResFileName = 'LFW_pairs_patch*';
        lfwpeopleResFileName = 'LFW_people_patch*';
        verificationResFileName = 'verification_patch*';
        verificationImagesFilePath = '../data/deepId_full/CFW_PubFig_SUFR/images_verification.txt';           
end