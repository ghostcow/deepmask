function DETS = runfacedet(I,imgpath)
    root=fullfile(fileparts(which(mfilename)), 'OpenCV_ViolaJones');
    detectorConf=fullfile(root,'haarcascade_frontalface_alt.xml');
    
    detector = cv.CascadeClassifier(detectorConf);
    boxes = detector.detect(I, ...
                            'ScaleFactor',  1.1, ...
                            'MinNeighbors', 2, ...
                            'MinSize', [40, 40]);

    
    if size(boxes,2) > 0
        T=zeros(size(boxes,2)+1,4);
        T(1,1)=size(boxes,2);
        T(2:end,:)=reshape(cell2mat(boxes),[4,size(boxes,2)])';
        
        widthVec=T(2:end,3);
        heightVec=T(2:end,4);
        T(2:end,3)=T(2:end,2);
        T(2:end,4)=heightVec+T(2:end,3);
        T(2:end,2)=widthVec+T(2:end,1);
        
        BB=T(2:end,:)';
        DETS=[(BB(1,:)+BB(2,:))/2 ; (BB(3,:)+BB(4,:))/2 ; (BB(2,:)-BB(1,:))/2];
        DETS(4,:)=1;
    else
        DETS=[];
    end
