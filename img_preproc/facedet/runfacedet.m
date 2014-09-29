function DETS = runfacedet(I,imgpath)

tmppath=tempname;
pgmpath=[tmppath '.pgm'];
if nargin<2
    detpath=[tmppath '.vj'];
else
    detpath=[imgpath '.vj'];
end

if ~exist(detpath, 'file')
    % run this only if the detections file doesn;t exist from previous run
    imwrite(I,pgmpath);

    root=fullfile(fileparts(which(mfilename)), 'OpenCV_ViolaJones');
    system(sprintf('%s/Release/OpenCV_ViolaJones %s/haarcascade_frontalface_alt.xml "%s" "%s"',root,root,pgmpath,detpath));
    delete(pgmpath);
end

DETS=readfacedets(detpath);
if nargin<2
    delete(detpath);
end
