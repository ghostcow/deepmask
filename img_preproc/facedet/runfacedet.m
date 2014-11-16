function DETS = runfacedet(I,imgpath)
    root=fullfile(fileparts(which(mfilename)), 'OpenCV_ViolaJones');
    detectorConf=fullfile(root,'haarcascade_frontalface_alt.xml');
    
    detector = cv.CascadeClassifier(detectorConf);
    if (size(I, 3) ~= 1)
        gr = rgb2gray(I); %cv.cvtColor(I, 'RGB2GRAY');
    else
        gr = I;
    end
    boxes = detector.detect(gr, ...
                            'ScaleFactor',  1.1, ...
                            'MinNeighbors', 3, ...
                            'Flags', 1, ... % opencv CV_HAAR_DO_CANNY_PRUNING flag
                            'MinSize', [40, 40]);

    
    if size(boxes,2) > 0
        T=reshape(cell2mat(boxes),[4,size(boxes,2)]);
        DETS=[T(1,:)+T(3,:)/2 ; 
              T(2,:)+T(4,:)/2 ; 
              T(3,:)/2 ; 
              ones(1,size(T(2,:),2))];
    else
        DETS=[];
    end
