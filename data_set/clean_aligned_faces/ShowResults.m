resultFilePath = './data/images_unknown_results_NLL.txt';
targetClass1Dir = '/media/data/datasets/CFW/filtered_aligned_network/face';
targetClass2Dir = '/media/data/datasets/CFW/filtered_aligned_network/background';

class1Th = 0.8;
class2Th = 0.8;
fid = fopen(resultFilePath);
C = textscan(fid, '%s %f %f', 'delimiter', ',');
fclose(fid);

imagePaths = C{1};
nImages = length(imagePaths);
class1Probs = C{2};
class2Probs = C{3};

figure('Name', 'Face');
a_faces = tight_subplot(9,9,[.01 .03],[.1 .01],[.01 .01]) ;
figure('Name', 'Background');
a_bg = tight_subplot(9,9,[.01 .03],[.1 .01],[.01 .01]) ;

iImageFace = 1;
iImageBg = 1;
ranIndices = randperm(nImages);
for iImage = ranIndices
   imagePath =  imagePaths{iImage};
   
   if (class1Probs(iImage) > class1Th)
       if (iImageFace > 9*9)
           continue;
       end
       axes(a_faces(iImageFace)); imshow(imagePath);
       iImageFace = iImageFace + 1;
   end   
   if (class2Probs(iImage) > class2Th)
       if (iImageBg > 9*9)
           continue;
       end       
       axes(a_bg(iImageBg)); imshow(imagePath);
       iImageBg = iImageBg + 1;
   end
   if ((iImageFace > 9*9) && (iImageBg > 9*9))
       break;
   end
end
