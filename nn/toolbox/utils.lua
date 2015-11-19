require 'sys'
require 'string'
local tds = require 'tds'

-- calculates center mass xcm,ycm of grayscale mask in DHW format
function getCenterMass(input)
    local M = torch.sum(input)
    local preWeight = torch.ones(input:size())
    local xWeights = torch.cumsum(preWeight, 3):typeAs(input)
    local yWeights = torch.cumsum(preWeight, 2):typeAs(input)
    local xSum = torch.sum(torch.cmul(input, xWeights))
    local ySum = torch.sum(torch.cmul(input, yWeights))
    return xSum/M, ySum/M
end

function checkBoundaries(x1, y1, x2, y2, img)
    --[[
        check if crop box fits in image around obj center of mass
        img is the image annotation object (from MSCOCO)
        assume x1<x2, y1<y2
        ]]
    if x1<1 or x2>img.width
            or y1<1 or y2>img.height then
        return false
    else
        return true
    end
end

function freeTable(t)
    if torch.type(k) == 'tds.Hash' or type(k) == 'table' then
        for k,_ in pairs(t) do
            t[k] = nil
            freeTable(k)
        end
    end
end

--[[ reshape and stack a table of tensors with identical dimensions
	can be used to create batch of images
	{CxHxW, CxHxW, CxHxW} -> 3xCxHxW
]]
function torch.vstack(intable)
    local cattable = {}
    local dim
    for _,v in pairs(intable) do
        dim = torch.LongStorage(v:dim()+1):fill(1)
        for i=1,v:dim() do dim[i+1]=v:size(i) end
        table.insert(cattable, v:view(dim))
    end
    return torch.cat(cattable, 1)
end

function time_func(msg, f, ...)
    sys.tic()
    local output = f(...)
    print(msg .. tostring(sys.toc()))
    return output
end

function getClasses()
    -- generate class to classname table for the confusion matrix
    local categories = torch.load(paths.concat(opt.dataPath, 'categories.tds.t7'))
    local classes = {}
    classes[81] = 'background'
    for k,_ in pairs(categories) do
        classes[k] = categories[k].name
    end
    return classes
end

function sizeTrain()
    return opt.batchSize * opt.epochSize
end