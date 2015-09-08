local gm = require 'graphicsmagick'

function wolfson02_scratch(imgpath)
    local out = gm.Image()
    imgpath = imgpath:gsub("/home/adampolyak","/a/home/cc/students/cs/adampolyak")
    out:load(imgpath, 100, 100):size(100, 100)
    out = out:toTensor('float','I','DHW')
    return out
end

function wolfson02_cifar(imgpath)
    local out = gm.Image()
    imgpath = imgpath:gsub("/home/adampolyak","/a/home/cc/students/cs/adampolyak")
    out:load(imgpath, 32, 32):size(32, 32)
    out = out:toTensor('float','RGB','DHW')
    return out
end

function wolfson03_scratch()
    local out = gm.Image()
    imgpath = imgpath:gsub("/home/adampolyak/datasets","/a/home/cc/students/cs/adampolyak/datasets03")
    out:load(imgpath, 100, 100):size(100, 100)
    out = out:toTensor('float','I','DHW')
    return out
end

function wolfson03_cifar()
    local out = gm.Image()
    imgpath = imgpath:gsub("/home/adampolyak/datasets","/a/home/cc/students/cs/adampolyak/datasets03")
    out:load(imgpath, 32, 32):size(32, 32)
    out = out:toTensor('float','RGB','DHW')
    return out
end

function pcwolf111_scratch(imgpath)
    local out = gm.Image()
    imgpath = imgpath:gsub("/a/home/cc/students/cs/adampolyak", "/home/adampolyak")
    out:load(imgpath, 100, 100):size(100, 100)
    out = out:toTensor('float','I','DHW')
    return out
end

function pcwolf111_cifar(imgpath)
    local out = gm.Image()
    imgpath = imgpath:gsub("/a/home/cc/students/cs/adampolyak", "/home/adampolyak")
    out:load(imgpath, 32, 32):size(32, 32)
    out = out:toTensor('float','RGB','DHW')
    return out
end