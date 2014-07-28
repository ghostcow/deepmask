function DETS = runfacedet(I,imgpath)

tmppath=tempname;
pgmpath=[tmppath '.pgm'];
if nargin<2
    detpath=[tmppath '.vj'];
else
    detpath=[imgpath '.vj'];
end

imwrite(I,pgmpath);

root=[fileparts(which(mfilename)) '/OpenCV_ViolaJones'];
system(sprintf('%s/Release/OpenCV_ViolaJones %s/haarcascade_frontalface_alt.xml %s %s',root,root,pgmpath,detpath));

DETS=readfacedets(detpath);
delete(pgmpath);
if nargin<2
    delete(detpath);
end
