function warped_image = Part5_piecewise_affine(current_shape, base_shape, input_image)
% Piecewise affine warp
% It is based on nearest neighbour interpolation
% Output: the warped warped shape
% The steps are the following:
% For all triangles
% Step 1: for each triangle in base_shape and current_shape, use the
% coordinates of their vertices to compute A
% Step 2: for each triangle find the pixels z within
% Step 3: Compute target coordinates 
% Step 4: Copy pixels (nearest neighbour)

% based on warp_2d from Octaam. Thank you for this! 

triangles = delaunay(base_shape(:,1), base_shape(:,2));
resolution = size(input_image);
base_texture = create_texture_base(base_shape, triangles, resolution);
num_of_triangles = size(triangles, 1);
A = zeros(1,6); % affine transformation for each triangle

% warped image
warped_image = zeros(resolution(1), resolution(2));

for t = 1:num_of_triangles
    %% Step 1:
    % Coordinates of the three vertices of each triangle in base shape
    U = base_shape(triangles(t,:),1) - 1;
    V = base_shape(triangles(t,:),2) - 1;
    
    % Coordinates of the three vertices of the corresponding triangle in current shape
    X = current_shape(triangles(t,:),1) - 1;
    Y = current_shape(triangles(t,:),2) - 1;
    
    % Compute A from the U,V and X,Y
    denominator = (U(2) - U(1)) * (V(3) - V(1)) - (V(2) - V(1)) * (U(3) - U(1));
    
    A(1) = X(1) + ((V(1) * (U(3) - U(1)) - U(1)*(V(3) - V(1))) * (X(2) - X(1)) + (U(1) * (V(2) - V(1)) - V(1)*(U(2) - U(1))) * (X(3) - X(1))) / denominator;
    A(2) = ((V(3) - V(1)) * (X(2) - X(1)) - (V(2) - V(1)) * (X(3) - X(1))) / denominator;
    A(3) = ((U(2) - U(1)) * (X(3) - X(1)) - (U(3) - U(1)) * (X(2) - X(1))) / denominator;
    
    A(4) = Y(1) + ((V(1) * (U(3) - U(1)) - U(1) * (V(3) - V(1))) * (Y(2) - Y(1)) + (U(1) * (V(2) - V(1)) - V(1)*(U(2) - U(1))) * (Y(3) - Y(1))) / denominator;
    A(5) = ((V(3) - V(1)) * (Y(2) - Y(1)) - (V(2) - V(1)) * (Y(3) - Y(1))) / denominator;
    A(6) = ((U(2) - U(1)) * (Y(3) - Y(1)) - (U(3) - U(1)) * (Y(2) - Y(1))) / denominator;
    
    
    %% Step 2
    % Get coordinates of all pixels within each triangle
    [v, u] = find(base_texture == t);
    
    if (~isempty(v) && ~isempty(u))
        
        ind_base = v + (u-1) * resolution(1);
        
        %% Step 3
        v = v - 1; u = u - 1;
        warped_x = A(1) + A(2) .* u + A(3) .* v + 1;
        warped_y = A(4) + A(5) .* u + A(6) .* v + 1;
        
        % round the values
        ind_warped = round(warped_y) + (round(warped_x)-1) * size(input_image, 1);
        
        %% Step 4
        % Find pixel coordinates in warped image
        max_ind_warped = size(input_image, 2)*size(input_image, 1);
        list = (ind_warped > 0 & ind_warped <= max_ind_warped);
        warped_image(ind_base(list)) = input_image(ind_warped(list));
        
    end
    
end

end

function base_texture = create_texture_base(vertices, triangles, resolution)
    % base_texture to warp image
    base_texture = zeros(resolution(1), resolution(2));

    for i=1:size(triangles,1)
        % vertices for each triangle
        X = vertices(triangles(i,:),1);
        Y = vertices(triangles(i,:),2);
        % mask for each traingle
        mask = poly2mask(X,Y,resolution(1), resolution(2)) .* i;
        % the complete base texture
        base_texture = max(base_texture, mask);
    end
end