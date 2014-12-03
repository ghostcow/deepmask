mainDir = 'faceleb_profile_images';
images = dir(fullfile(mainDir, '*.jpg'));

for iImage = 1:length(images)
   imageName = images(iImage).name;
   figName = imageName(1:end-4);
   num = str2double(figName(1:5));
   
   figDir = fullfile(mainDir, figName);
   if ~exist(figDir, 'dir')
       mkdir(figDir);
   end
   newImageName = [figName '_0000.jpg'];
   movefile(fullfile(mainDir, imageName), ...
       fullfile(figDir, newImageName));
end