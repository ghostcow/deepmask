addpath('../../utils');

%%
mainDir = '/media/data/datasets/';
pubfigDir = fullfile(mainDir, 'pubfig', 'images');
cfwDir = fullfile(mainDir, 'CFW', 'images');
mapNamesPath = fullfile(mainDir, 'pubfig', 'MapToCfwNames.csv');

%%
fid = fopen(mapNamesPath);
C = textscan(fid, '%s %s', 'Delimiter', ',');
fclose(fid);
pubFigNames = C{1}(2:end);
cfwNames = C{2}(2:end);
numDups = 0;
for iPerson = 1:numel(pubFigNames)
   pubFigName =  pubFigNames{iPerson};
   cfwName =  cfwNames{iPerson};
   if ~strcmp(cfwName, '---')
       cfwDir = fullfile(cfwDir, cfwName);
       cfwImages = dir(fullfile(cfwDir, '*.jpg'));
       cfwMd5 = cell(1, numel(cfwImages));
       for iImage = 1:numel(cfwImages)
          imagePath = fullfile(cfwDir, cfwImages(iImage).name);
          fid = fopen(imagePath);
          hash = DataHash(fread(fid));
          fclose(fid);
          cfwMd5{iImage} = hash;
       end
       
       pubfigDir = fullfile(pubfigDir, pubFigName);
       pubfigImages = dir(fullfile(pubfigDir, '*.jpg'));
       for iImage = 1:numel(pubfigImages)
          imagePath = fullfile(pubfigDir, pubfigImages(iImage).name);
          fid = fopen(imagePath);
          hash = DataHash(fread(fid));
          fclose(fid);
          
          a = ismember(cfwMd5, hash);
          aa = find(a);
          if ~isempty(aa)
             dupFilePath = fullfile(cfwDir, cfwImages(aa(1)).name);
             fprintf('duplicate file - %s,%s\n', imagePath, dupFilePath);
             numDups = numDups + 1;
             delete(dupFilePath);
          end
       end
   end
end