require 'torch'
local ffi = require 'ffi'
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'math'
require 'utils'
local tds = require 'tds'
local gm = require 'graphicsmagick'

local dataset = torch.class('torch.CocoDataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
     Coco dataset loader
]],
    {name="dataPath",
     type="string",
     help="Path to data tds"},

    {name="splitName",
     type="string",
     help="Data split name"},

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
    local ds = paths.concat(self.dataPath, self.splitName)
    self.instances  = torch.load(ds .. '.instances.tds.t7')
    self.imgs       = torch.load(ds .. '.imgs.tds.t7')
    self.img2inst   = torch.load(ds .. '.img2ann.tds.t7')
    self.class2inst = torch.load(ds .. '.class2instance.tds.t7')
end

function dataset:loadImg(imgName)
    return image.load(self.cocoImagePath .. '/' .. imgName)
end

function dataset:samplePositive(class)
    --loads image and annotation and crops / extra: flip,scale and shift
    local sample
    while not sample do
        local inst = self:getInstanceByClass(class)
        sample = sampleInstance(inst)
    end
    return sample
end

function dataset:getInstanceByClass(class)
    local list = self.class2inst[class]
    return list[torch.random(1,list:size(1))]
end

local function sampleInstance(inst)
    local ann = self.instances[inst]
    local img_ann = self.imgs[inst.image_id]
end

function dataset:sampleNegative()
    -- load img and loop until you crop a negative sample
    local index = torch.random()
end

-- samples a single image (negative or positive)
function dataset:sample()
    if torch.uniform() < self.negativeRatio then
        return self:sampleNegative()
    else
        local class = torch.random(1,80)
        return self:samplePositive(class)
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

function dataset:sizeTrain()
    return opt.batchSize * opt.epochSize
end
