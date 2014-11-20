mainDir = '/media/data/datasets/WLFDB'; 
alignedImagesDir = fullfile(mainDir, 'aligned_deepid');
figDir = fullfile(alignedImagesDir, '2552_Timothy_Olyphant');

im1 = imread(fullfile(figDir, '2552Timothy_Olyphant_0036.jpg'));
im2 = imread(fullfile(figDir, '2552Timothy_Olyphant_0038.jpg')); % 136, 138, 102, 113, 1
cedd1 = CEDD(im1);
cedd2 = CEDD(im2);
disp('different images');
distTanimoto = Tanimoto(cedd1, cedd2)

disp('duplicate images');
im2 = imread(fullfile(figDir, '2552Timothy_Olyphant_0138.jpg')); % 136, 138, 102, 113, 1
cedd2 = CEDD(im2);
distTanimoto = Tanimoto(cedd1, cedd2)