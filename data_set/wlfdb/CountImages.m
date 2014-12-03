clear variables;
mainDir = 'faceleb_profile_images';
images = dir(fullfile(mainDir, '*.jpg'));
nProgilePics = length(images);

fprintf('#persons with profile image=%d\n', nProgilePics);
imagesPerPerson = zeros(1,40000);
for iImage = 1:length(images)
   imageName = images(iImage).name;
   figName = imageName(1:end-4);
   num = str2double(figName(1:5));
   imagesPerPerson(num) = 1;
   
   figImages = dir(fullfile(mainDir, figName, '*.jpg'));
   imagesPerPerson(num) = imagesPerPerson(num) + length(figImages);
end
nProfileAndMorePics = sum(imagesPerPerson > 1);
fprintf('#persons with profile pic and more=%d\n', nProfileAndMorePics);

figDirs = dir(mainDir);
figDirs = figDirs(3:end);
fprintf('#persons without profile image=%d\n', length(figDirs) - nProfileAndMorePics);
for iDir = 1:length(figDirs)
    num = str2double(figDirs(iDir).name(1:5));
    if (imagesPerPerson(num) == 0)
        imagesPerPerson(num) = length(dir(fullfile(mainDir, figDirs(iDir).name, '*.jpg')));
    end
end
