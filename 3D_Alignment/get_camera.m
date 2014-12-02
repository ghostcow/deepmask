function P = get_camera(landmarks2D, landmarks3D, cov_mat)    
    % construct X3d
    no_samples=size(landmarks2D,2);
    
    x2d=landmarks2D(:);
    X3d=zeros(no_samples*2, 8);
    for i=1:no_samples
        curr_ind = 2*(i-1)+1;
        X3d(curr_ind:curr_ind+1,:)=[landmarks3D(:,i)' 1 zeros(1,4); ...
                                    zeros(1,4) landmarks3D(:,i)' 1];
    end
    
    % get camera
    P = lscov(X3d, x2d, cov_mat);
    P = reshape(P,4,2)';
end

