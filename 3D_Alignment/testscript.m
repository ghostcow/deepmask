XYZ = rand(3,1000);
P = rand(2,4);
XYZ(4,:) = 1;

xy = P*XYZ;
%xyone = bsxfun(@mult,xyone,1./xyone(3,:));
%xy(3,:) = 1;
rP = get_camera(xy,XYZ(1:3,:) , eye(2000));
P./rP
[rXYZ,PIP, PIP2] = get_residual(xy, XYZ(1:3,:), rP);
rXYZ(4,:) = 1;
xy2 = P*rXYZ;
PIP(4,:) = 1;
xy3 = P*PIP;
PIP2(4,:) = 1;
xy4 = P*PIP2;

std(xy2./xy,[],2)
std(xy3./xy,[],2)
std(xy4./xy,[],2)

%newxy = rXYZ(1:2,:)*10;
%newxy = xy+randn(size(xy,1),size(xy,2))/1000;
%tform = fitgeotrans(xy',newxy','pwl');

