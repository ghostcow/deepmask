inputImagesDir = '/media/data/datasets/CFW/filtered_aligned_network/temp';
targetImagesDir = '/media/data/datasets/CFW/filtered_aligned_network/byFigure';

inputDirs = dir(inputImagesDir);
inputDirs = inputDirs(3:end);
for iDir = 1:length(inputDirs)
   dirName = inputDirs(iDir).name;
   images = dir(fullfile(inputImagesDir, dirName, '*.jpg'));
   for iImage = 1:length(images)
      imageName = images(iImage).name;
      k = strfind(imageName, '.');
      if (length(k) == 2)
        figureName = imageName(1:(k(end-1)-1));
        imageNum = imageName((k(end-1)+1):(k(end)-1));
      elseif (length(k) == 3)
        figureName = imageName(1:(k(end-2)-1));
        imageNum = imageName((k(end-2)+1):(k(end)-1));          
      else
          error('something is wrong...');
      end
      
      targetImageDir = fullfile(targetImagesDir, figureName);
      if ~exist(targetImageDir, 'dir')
          mkdir(targetImageDir);
          disp(figureName);
      end
      targetPath = fullfile(targetImageDir, [imageNum '.jpg']);
      if ~exist(targetPath, 'file')
        copyfile(fullfile(inputImagesDir, dirName, imageName), targetPath);
      end
   end
end