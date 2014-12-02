function [rXYZ, P_image_plane, P_image_plane2] = get_residual(x2d, X3d, P)
%get_residual Returns nearest point for X3d on rays that pass the image 
%              plane. Instead of using the camera centre, we use 2
%              different psuedo inverse matricies.
    
    %% calculating camera image plane using psuedo inverse
    temp_P=[P; 0 0 0 1];
    psuedo_inv_P = ((temp_P*temp_P')\temp_P)'; % same as temp_P*temp_P'*x=temp_P
    
    temp_P2 = [P;0 0 0 1; 1 1 1 1];
    psuedo_inv_P2 = inv(temp_P2);
    
    % we don't devide each column by the fourth member - because by adding
    % the lines [0 0 0 1] to P(or P2), we keep this to 1.
    P_image_plane=psuedo_inv_P*[x2d; ones(1,size(x2d,2))];
    P_image_plane=P_image_plane(1:3,:);
    P_image_plane2=psuedo_inv_P2*[x2d; ones(2,size(x2d,2))];
    P_image_plane2=P_image_plane2(1:3,:);
    
    %% find nearest point on camera ray    
    % this the same as t=dot(x1-x0, x2-x1)/(norm(x2-x1,2)^2) - in matrix
    % notation
    nominator=dot(P_image_plane-X3d, P_image_plane2-P_image_plane);
    denominator=sum((P_image_plane2-P_image_plane).^2,1);
    t=-1*(nominator./denominator);
    % this is the same as rXYZ(i,:)=x1+t*(x2-x1) - in matrix notation
    rXYZ=P_image_plane+bsxfun(@times,P_image_plane2-P_image_plane,t);
end

