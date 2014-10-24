addpath('../utils');

%%
mainDir = '/media/data/datasets/';
imagesDir = fullfile(mainDir, 'CFW', 'images');
targetImagesDir = fullfile(mainDir, 'CFW', 'filtered_aligned_network', 'byFigure'); % where to delete the dup images

%%
personDirs = dir(imagesDir);
personDirs = personDirs(3:end);
for iPerson = 1090:numel(personDirs)
   personName =  personDirs(iPerson).name;
   fprintf('%d - %s\n', iPerson, personName);
   images = dir(fullfile(imagesDir, personName, '*.jpg'));
   hashes = cell(1, numel(images));
   for iImage = 1:numel(images)
      imagePath = fullfile(imagesDir, personName, images(iImage).name);
      fid = fopen(imagePath);
      hash = DataHash(fread(fid));
      fclose(fid);
      
      hashes{iImage} = hash;
      a = ismember(hashes(1:(iImage-1)), hash);
      aa = find(a);
      if ~isempty(aa)
          % look for one of the aligned images
          alignedImagePath = fullfile(targetImagesDir, personName, images(iImage).name);
          if exist(alignedImagePath, 'file')
              fprintf('dup images : %s,%s\n', images(iImage).name, images(aa(1)).name);
          end
      end
   end
end