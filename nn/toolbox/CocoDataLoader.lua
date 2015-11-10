require 'torch'
local ffi = require 'ffi'
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'math'
require 'tds'
local gm = require 'graphicsmagick'

local dataset = torch.class('torch.dataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
     Coco dataset loader
]],
    {name="dataPath",
     type="string",
      help="Path to data tds"},

    {name="cocoImagePath",
        type="string",
        help="Path to data tds"},

    {name="negativeRatio",
        help="Ratio for negative from batch",
        default = 0.5},
}

function dataset:__init(...)

    -- argcheck
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end

    -- load ms coco index files
    local ds = self.dataPath
    self.img2ann = torch.load(ds .. '.img2ann.tds.t7')
    self.imgs = torch.load(ds .. '.imgs.tds.t7')
    self.ann = torch.load(ds .. '.ann.tds.t7')

    -- map from class to list of annotations
end

function dataset:loadImg(imgName)
    return image.load(self.cocoImagePath .. '/' .. imgName)
end

function dataset:samplePositive(img, ann)
    --loads image and annotation and crops / extra: flip,scale and shift
end

function dataset:sampleNegative()
    -- load img and loop until you crop a negative sample
    local index = torch.random()
end

--TODO: change to load by random class first
function dataset:getByClass(class)
    local list = self.class2ann[class]
    return list[torch.random(1,list:size(1))]
end

-- samples a single image (negative or positive)
function dataset:sample()
    if torch.uniform() < self.negativeRatio then
        local class = 0
        return self:sampleNegative()
    else
        local class = torch.random(1,80)
        -- TODO: loop here until you get a good annotation
        local ann = self:getByClass(class)

        return self:samplePositive(ann)
    end
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(dataTable, maskTable, labelTable)
    local data, masks, labels
    local quantity = #dataTable

    data = torch.Tensor(quantity, 3, 224, 224)
    masks = torch.Tensor(quantity, 1, 224, 224)
    labels = torch.Tensor(quantity)

    for i=1,quantity do
        data[i]:copy(dataTable[i])
        masks[i]:copy(maskTable[i])
        labels[i] = labelTable[i]
    end

    return data, masks, labels
end

-- TODO: don't forget about sampling procedure
function dataset:get(batchSize, branch)
    local dataTable, maskTable, labelTable = {}, {}, {}

    if branch == 1 then
        -- sample masks, only positive here
        for _ = 1,batchSize do
            local img, mask, label = self:sample()
            table.insert(dataTable, img)
            table.insert(maskTable, mask)
            table.insert(labelTable, label)
        end
    else
        -- sample scores only
    end

    local data, masks, labels = tableToOutput(dataTable, maskTable, labelTable)
    return data, masks, labels, branch
end

