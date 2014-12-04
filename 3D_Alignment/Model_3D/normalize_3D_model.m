function xy_3d = normalize_3D_model(xyz_3d, scale)
    % normalize x,y coordinates from the 3D model
    xy_3d = xyz_3d([1 3],:);
    xy_3d_min = min(xy_3d, [], 2);
    xy_3d_max = max(xy_3d, [], 2);

    xy_3d = bsxfun(@minus, xy_3d, xy_3d_min);
    xy_3d = bsxfun(@times, xy_3d, 1 ./ (xy_3d_max - xy_3d_min));

    xy_3d = bsxfun(@times, xy_3d, scale);
    % vertical flip of  x,y coordinates
    xy_3d(1,:) = scale(1) - xy_3d(1,:); 
    xy_3d(2,:) = scale(2) - xy_3d(2,:);  
end

