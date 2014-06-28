function [detections, landmarks, err_imgs] = align_lfw(imgDir, opts)
%DETECT_ALL_LFW Summary of this function goes here
%   Detailed explanation goes here
    imgList = get_file_list(imgDir);    
    numImg = numel(imgList);
    
    detections = zeros(numImg, 4, 1);
    landmarks = zeros(numImg, 2, 9);
    
    err_imgs = cell(numImg, 1);
    err_counter = 0;
    
    for i=1:numImg
        try
            [dets, pts, aligned] = align_face(opts, imgList{i});
            detections(i,:,:) = dets;
            landmarks(i,:,:) = pts;
            [f_dir, f_basename, f_ext] = fileparts(imgList{i});
            aligned_img_name = fullfile(f_dir, strcat(f_basename,'_aligned',f_ext))
            imwrite(aligned, aligned_img_name);
        catch
            err_imgs{err_counter+1} = imgList{i};
            err_counter = err_counter + 1;
        end
    end

end

