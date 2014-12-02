function points = get_3D_model()
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
    points=points';
end

