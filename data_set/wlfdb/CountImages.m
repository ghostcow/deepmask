clear variables;
mainDir = 'faceleb_profile_images';
images = dir(fullfile(mainDir, '*.jpg'));
fprintf('#persons with profile image=%d\n', length(images));
noImages = 0;
for iImage = 1:length(images)
   imageName = images(iImage).name;
   figName = imageName(1:end-4);
   figImages = dir(fullfile(mainDir, figName, '*.jpg'));
   if isempty(figImages)
       noImages = noImages + 1;
   end
end
fprintf('#persons with no more images=%d\n', noImages);
fprintf('#persons with more than 1 image=%d\n', length(images) - noImages);

% figDirs = dir(mainDir);
% figDirs = figDirs(3:end);
% for iDir = 1:length(figDirs)
%    images = dir(fullfile(mainDir, figDirs(iDir).name));
%    if isempty(images)
%        rmdir(fullfile(mainDir, figDirs(iDir).name));
%    end
% end