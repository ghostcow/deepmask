require 'torch'
local argcheck = require 'argcheck'
require 'sys'
require 'math'
require 'utils'
local tds = require 'tds'; tds.hash.__ipairs = tds.hash.__pairs
local gm = require 'graphicsmagick'

local dataset = torch.class('torch.CocoDataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[
     Coco dataset loader
]],
    {name="splitName",
        type="string",
        help="Data split name"},

    {name="dataPath",
     type="string",
     help="Path to annotations dir (tds, masks)"},

    {name="cocoImagePath",
     type="string",
     help="Path to COCO images"},

    {name="negativeRatio",
     help="Ratio for negative from batch",
     default = 0.5},
}

function dataset:__init(...)

    -- argcheck
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end
    -- some paths
    self.ann_dir = paths.concat(self.dataPath, self.splitName .. '_anns')
    self.image_dir = paths.concat(self.cocoImagePath, self.splitName)
    -- load ms coco index files
    local ds = paths.concat(self.dataPath, self.splitName)
    self.instances  = torch.load(ds .. '.instances.tds.t7')
    self.imgs       = torch.load(ds .. '.imgs.tds.t7')
    self.img2inst   = torch.load(ds .. '.img2ann.tds.t7')
    self.class2inst = torch.load(ds .. '.class2instance.tds.t7')
    self.imgidx     = torch.load(ds .. '.imgidx.tds.t7')
    -- image buffers
    self.img = gm.Image()
    self.raw_mask = gm.Image()
end

function dataset:loadImg(imgName)
    return image.load(self.cocoImagePath .. '/' .. imgName)
end

function dataset:samplePositive()
    -- returns a sample with an object in it,
    -- randomized over class, augmentations:
    -- scale deformation, translation shift, flipping.
    local class = torch.random(1,80)
    local patch, mask, label
    while not patch do
        local inst = self:getInstanceByClass(class)
        patch, mask, label = self:sampleInstance(inst)
    end
    return patch, mask, label
end

function dataset:getInstanceByClass(class)
    local list = self.class2inst[class]
    return list[torch.random(1,#list)]
end

function dataset:sampleInstance(inst)
    local ann = self.instances[inst]
    local img_ann = self.imgs[inst.image_id]

    local img_path = paths.concat(self.image_dir, img_ann.filename)
    local im = self.img:load(img_path):toTensor('float', 'RGB', 'DHW', true)
    local raw_mask_path = paths.concat(self.ann_dir, img_ann.id, ann.id .. '.png')
    local raw_mask = self.raw_mask:load(raw_mask_path):toTensor('float', 'I', 'DHW', true)

    if ann.iscrowd == 0 then
        local xcm, ycm = getCenterMass(raw_mask)
        -- get long edge
        local long_edge = math.max(ann.bbox[3], ann.bbox[4])
        -- check to see if obj is at an acceptable scale
        if long_edge >= 15 and long_edge <= 304 then

            -- scale deformation
            local scale = 2^(torch.random(-1,1)*0.25)
            -- translation shift
            local xshift = torch.random(-1,1)*16
            local yshift = torch.random(-1,1)*16
            -- crop mask (label) and image patch
            local width = scale * (7/4 * long_edge)
            local height = width
            -- scale the shift too
            local x1 = (xcm-width/2+1/2)  + width/224*xshift
            local y1 = (ycm-height/2+1/2) + height/224*yshift
            local x2 = x1 + width  - 1
            local y2 = y1 + height - 1

            -- if crop is within bounds, return patch mask label
            -- TODO: add feature to pad crops that aren't fully contained
            if checkBoundaries(x1, y1, x2, y2, img_ann) then

                -- crop image
                local patch = image.crop(im, x1, y1, x2, y2)
                local mask = image.crop(raw_mask, x1, y1, x2, y2)
                -- horizontal flip
                if torch.random(0,1) == 1 then
                    patch = image.hflip(patch)
                    mask = image.hflip(mask)
                end
                -- scale to 224x224
                patch = image.scale(patch, 224, 224)
                mask = image.scale(mask, 224, 224)
                return patch, mask, ann.category_id
            end
        end
    end
    -- if the instance can't be cropped
    return nil
end

function dataset:sampleNegative()
    -- load img and loop until you crop a negative sample
    local img_id = self.imgidx[torch.random(1,#self.imgidx)]
    local instances = self.img2inst[img_id]
    -- select scale
    local scale = 2^(torch.random(-4,6)*0.5)
    local x = torch.random(1, self.imgs[img_id].height * scale)
    local y = torch.random(1, self.imgs[img_id].width * scale)

    for _,v in pairs(instances) do
        local raw_mask = self.raw_mask:load(paths.concat(self.ann_dir, img_id, v .. '.png'))
                                      :toTensor('float', 'I', 'DHW')
        local xcm, ycm = getCenterMass(raw_mask)
        local long_edge = math.max(v.bbox[3], v.bbox[4])

        if torch.abs(xcm*scale-(x +112-1)) < 32
                and torch.abs(ycm*scale-(y +112-1)) < 32 then
            return nil
        end
        if long_edge < 256*1/scale and long_edge > 64*1/scale then
            return nil
        end
    end
    -- if the patch is far enough away (by location, scale) from all instances, crop scale and return
    local patch = image.crop(self.img:load(paths.concat(self.image_dir, self.imgs[img_id].filename))
                                     :crop(224,224,x,y)
                                     :toTensor('float', 'RGB', 'DHW'))
    return patch
end

-- samples a single image (negative or positive depending on branch)
function dataset:sample(branch)
    if branch == 1 then
        return self:samplePositive()
    else
        if torch.uniform() < self.negativeRatio then
            return self:sampleNegative()
        else
            return self:samplePositive()
        end
    end
end

-- converts a table/tds.vec of samples (and corresponding labels) to a clean tensor
local function tableToOutput(patchTable, maskTable, labelTable, branch)
    local patches, masks, labels
    local quantity = #patchTable

    patches = torch.Tensor(quantity, 3, 224, 224)
    labels = torch.Tensor(quantity)

    for i=1,quantity do
        patches[i]:copy(patchTable[i])
        labels[i] = labelTable[i]
    end

    if branch == 1 then
        masks = torch.Tensor(quantity, 1, 224, 224)
        for i=1,quantity do
            masks[i]:copy(maskTable[i])
        end
        return patches, masks, labels
    else
        return patches, labels
    end
end

-- TODO: change from table to tds if necessary
function dataset:get(batchSize, branch)
    local patchTable, maskTable, labelTable = {}, {}, {}

    if branch == 1 then
        -- sample masks, only positive here
        for _=1,batchSize do
            local img, mask, label = self:sample(branch)
            table.insert(patchTable, img)
            table.insert(maskTable, mask)
            table.insert(labelTable, label)
        end
        local patches, masks, labels = tableToOutput(patchTable, maskTable, labelTable, branch)
        return patches, masks, labels, branch
    else
        -- sample scores only
        for _=1,batchSize do
            local patch, label = self:sample(branch)
            table.insert(patchTable, patch)
            table.insert(labelTable, label)
        end
        local patches, labels = tableToOutput(patchTable, nil, labelTable, branch)
        return patches, labels, branch
    end


end

function dataset:sizeTrain()
    return opt.batchSize * opt.epochSize
end
