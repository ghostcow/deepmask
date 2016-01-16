require 'torch'
local argcheck = require 'argcheck'
require 'sys'
require 'math'
require 'utils'
local tds = require 'tds'
tds.hash.__ipairs = tds.hash.__pairs
local gm = require 'graphicsmagick'
require 'image'

local dataset = torch.class('torch.CocoDataLoader')

local initcheck = argcheck{
    pack=true,
    help=[[ COCO dataset loader
    ]],
    {name="splitName",
        type="string",
        help="Data split name"},

    {name="dataPath",
        type="string",
        help="Path to annotations dir (tds, masks)"},

    {name="imageDirPath",
        type="string",
        help="Path to COCO images"},

    {name="negativeRatio",
        type="number",
        default=0.5,
        help="Ratio for negative from batch"}
}

function dataset:__init(...)

    -- argcheck
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end
    -- some paths
    self.ann_dir = paths.concat(self.dataPath, self.splitName .. '_masks')
    self.image_dir = paths.concat(self.imageDirPath, self.splitName)
    -- load ms coco index files
    local prefix = paths.concat(self.dataPath, self.splitName)
    self.instances = torch.load(prefix .. '.instances.tds.t7')
    self.imgs      = torch.load(prefix .. '.imgs.tds.t7')
    self.imgidx    = torch.load(prefix .. '.imgidx.tds.t7')
    self.img2inst  = torch.load(prefix .. '.img2inst.tds.t7')
    self.class2cat = torch.load(prefix .. '.class2cat.tds.t7') -- here 'cat' means the category id (80 of 1,90 range)
    self.cat2class = torch.load(prefix .. '.cat2class.tds.t7') -- class means the class we train with (1,80 range)
    self.cat2inst  = torch.load(prefix .. '.cat2instance.tds.t7')

    -- ImageNet mean, from VGG Gist
    self.mean = {123.68, 116.779, 103.939}
end

-- converts a table/tds.vec of samples (and corresponding labels) to a clean tensor
local function tableToOutput(branch, labelTable, patchTable, maskTable)
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
            local patch, mask, label = self:sample(branch)
            table.insert(patchTable, patch)
            table.insert(maskTable, mask)
            table.insert(labelTable, label)
        end
        local patches, masks, labels = tableToOutput(branch, labelTable, patchTable, maskTable)
        return branch, labels, patches, masks
    else
        -- sample scores only
        for _=1,batchSize do
            local patch, label = self:sample(branch)
            table.insert(patchTable, patch)
            table.insert(labelTable, label)
        end
        local patches, labels = tableToOutput(branch, labelTable, patchTable)
        return branch, labels, patches
    end
end

-- samples a single image (negative or positive depending on branch)
function dataset:sample(branch)
    if branch == 1 then
        return self:samplePositive(branch)
    else
        if torch.uniform() < self.negativeRatio then
            return self:sampleNegative()
        else
            return self:samplePositive(branch)
        end
    end
end

-- subtracts mean from input RGB image
local function subtractMean(im, mean)
    for i=1,3 do
        im[i]:csub(mean[i])
    end
    return im
end

local function preProcessPatch(im, mean)
  -- rescale the image and return it to range 0..255
  local im2 = image.scale(im, 224, 224, 'bilinear') * 255
  -- subtract imagenet mean
  local im3 = subtractMean(im2, mean)
  -- RGB2BGR
  return im3:index(1,torch.LongTensor{3,2,1})
end

local function preProcessMask(im)
  -- rescale the image
  return image.scale(im, 224, 224, 'bilinear')
end

function dataset:samplePositive(branch)
    -- returns a sample with an object in it,
    -- randomized over class, augmentations:
    -- scale deformation, translation shift, flipping.
    local class = torch.random(1,80)
    local inst, patch, mask, label

    while not patch do
        inst = self:getInstanceByClass(class)
        patch, mask, label = self:sampleInstance(inst)
    end

    if branch == 1 then
        return patch, mask, label
    else
        return patch, label
    end
end

function dataset:getInstanceByClass(class)
    local list = self.cat2inst[self.class2cat[class]]
    return list[torch.random(1,#list)]
end

function dataset:sampleInstance(inst)
    local ann = self.instances[inst]
    local img_ann = self.imgs[ann.image_id]

    if ann.iscrowd == 0 then

        local im = self:loadImg(img_ann.id)
        local raw_mask = self:loadMask(img_ann.id, ann.id)

        local xcm, ycm = getCenterMass(raw_mask)
        -- get long edge
        local long_edge = math.max(ann.bbox[3], ann.bbox[4])
        -- check to see if obj is at an acceptable scale
        if long_edge >= 26 and long_edge <= 304 then

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
                -- scale to 224x224 and return
                --return image.scale(patch, 224, 224), image.scale(mask, 224, 224), self.cat2class[ann.category_id]
                return preProcessPatch(patch, self.mean), preProcessMask(mask), self.cat2class[ann.category_id]
            end
        end
    end
    -- if the instance can't be cropped
    return nil
end

function dataset:sampleNegative()
    local rawPatch
    while rawPatch == nil do
        -- load img and loop until you crop a negative sample
        local imgId = self.imgidx[torch.random(1,#self.imgidx)]
        local instances = self.img2inst[imgId]

        -- select scale
        local scale = 2^(torch.random(-4,6)*0.5)

        -- random x,y coords from target image
        local x = torch.random(1, self.imgs[imgId].width * scale)
        local y = torch.random(1, self.imgs[imgId].height * scale)

        if checkBoundaries(x/scale, y/scale, (x+224-1)/scale, (y+224-1)/scale, self.imgs[imgId]) then

            -- determine if the patch is far enough away (by location, scale) from all instances
            local tooClose
            for _,instIdx in pairs(instances) do
                local instance = self.instances[instIdx]
                if instance.iscrowd == 0 and self:instanceTooClose(x,y,imgId,instIdx,instance,scale) then
                    tooClose = true
                    break
                end
            end
            if not tooClose then
                rawPatch = image.crop(self:loadImg(imgId),
                    x/scale, y/scale,
                    (x+224-1)/scale, (y+224-1)/scale)
            end
        end
    end
    -- scale and return. class 81 is background
    --return subtractMean(image.scale(rawPatch, 224, 224), self.mean), 81
    return preProcessPatch(rawPatch, self.mean), 81
end

function dataset:instanceTooClose(x, y, imgId, instIdx, instance, scale)
    local raw_mask = self:loadMask(imgId, instIdx)
    local xcm, ycm = getCenterMass(raw_mask)
    local long_edge = math.max(instance.bbox[3], instance.bbox[4])

    if torch.abs(xcm-(x+112-1)/scale) < 32/scale
            and torch.abs(ycm-(y+112-1)/scale) < 32/scale then
        return true
    end
    if long_edge < 256/scale and long_edge > 64/scale then
        return true
    end
    return false
end

function dataset:loadMask(imgId, instId)
    local width, height = self:getImageInfo(imgId)
    local rawMaskPath = paths.concat(self.ann_dir, imgId, instId .. '.png')
    return gm.Image(rawMaskPath, width, height):toTensor('float', 'I', 'DHW')
end

function dataset:loadImg(imgId)
    local width, height, name = self:getImageInfo(imgId)
    local imgPath = paths.concat(self.image_dir, name)
    return gm.Image(imgPath, width, height):toTensor('float', 'RGB', 'DHW')
end

function dataset:getImageInfo(imgId)
    return self.imgs[imgId].width, self.imgs[imgId].height, self.imgs[imgId].file_name
end

function dataset:sizeTrain()
    return opt.batchSize * opt.epochSize
end
