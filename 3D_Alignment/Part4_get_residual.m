function [r, P_image_plane] = Part4_get_residual(x2d, X3d, P)
%PART4_FORNTALIZATION Summary of this function goes here
%   Detailed explanation goes here
    
    %% find camera centre and projected points
     P_center=null([P; 0 0 0 1]);
     x2d=[x2d(1:2:end)'; x2d(2:2:end)']';
     P_image_plane=zeros(size(x2d,1),3);
     
%     for i=1:size(x2d,1)
%         v = x2d(i,:)'-P(:,3)-P(:,4);
%          P_image_plane(i,:)= [P(1:2,1:2)\v; 1];
%     end
    
    %% calculating camera image plane using psuedo inverse
    temp_P=[P; 0 0 0 1];
    psuedo_inv_P = ((temp_P*temp_P')\temp_P)'; % same as temp_P*temp_P'*x=temp_P
    for i=1:size(x2d,1)
        v=psuedo_inv_P*[x2d(i,:) 1]';
        P_image_plane(i,:)= v(1:3)/v(4);
    end
    
    %% add residual
    % calculate nearest point on line from 
    r=zeros(size(x2d,1),3);
    x2=P_center(1:3)';
    for i=1:size(x2d,1)
        x0=X3d(i,:);
        x1=P_image_plane(i,:);
        t=dot(x1-x0, x2-x1)/(norm(x2-x1,2)^2); 
        t=-1*t;

        r(i,:)=x1+t*(x2-x1);
        
        % calculate distrance for x-y axis
        % d = abs(cross(x2-x1,x0-x1))/abs(x2-x1);
    end      
end

