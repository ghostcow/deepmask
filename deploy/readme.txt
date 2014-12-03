installing all dependencies for the lua code to run properly :
sudo install-luajit+torch
sudo install-deps
sudo install-OpenBLAS (if neccessary)
sudo install-ccn

NOTE1: 
all scripts should be run under su, with sudo prefix

NOTE2: 
cutorch & cunn should installed by install-luajit+torch.
However, sometimes it doesn't happen so you should install them manually : 
sudo luarocks install cunn
sudo luarocks install cutorch

NOTE3:
The MATLAB code in img_preproc require the mexopencv package to be located in the same directory as the project main directory. You can have it by calling :
git clone https://github.com/adampolyak/mexopencv
