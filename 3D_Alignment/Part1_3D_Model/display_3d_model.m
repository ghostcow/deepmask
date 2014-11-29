function points = display_3d_model()	
	load 3D_model.mat
    load 3D_points.mat

    points=[nm_nose; 
            nm_left_eye; 
            nm_left_brow;   
            nm_right_eye;
            nm_right_brow;
            nm_left_top_mouth;
            nm_right_top_mouth;
            nm_bottom_mouth;
            nm_left_face(1,:); % fixed order issue
            flipud(nm_left_face(2:9,:));
            nm_right_face;
            ];

    i=1;
    j=size(points,1);

    % display 3d image with landmarks
    display_shape(shape,tl); 
    hold on; 
    plot3(points(i:j,1), points(i:j,2), points(i:j,3), 'r.');
    labels=cellstr(num2str([i:j]'));
    text(points(i:j,1), points(i:j,2), points(i:j,3), labels);