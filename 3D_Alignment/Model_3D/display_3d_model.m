function points = display_3d_model()	
	load 3D_model.mat
    points=get_3D_model();
    
    i=1;
    j=size(points,2);

    % display 3d image with landmarks
    display_shape(shape,tl); 
    hold on; 
    plot3(points(1,i:j), points(2,i:j), points(3,i:j), 'r.');
    labels=cellstr(num2str([i:j]'));
    text(points(1,i:j), points(2,i:j), points(3,i:j), labels);