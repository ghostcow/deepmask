function edges = get_image_edges(im)
    [height, width, ~] = size(im);

    horizonPoints = round(width/5):round(width/5):round(width*4/5);
    verticalPoints = round(height/7):round(height/7):round(height*6/7);

    edges = [1 1; 
            width 1; 
            width height; 
            1 height;
            horizonPoints', ones(length(horizonPoints), 1); ...
            horizonPoints', height*ones(length(horizonPoints), 1); ...
            ones(length(verticalPoints), 1), verticalPoints'; ...
            width*ones(length(verticalPoints), 1), verticalPoints'];
end

