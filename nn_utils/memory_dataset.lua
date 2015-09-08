require 'sys'
require 'torch'
local argcheck = require 'argcheck'

local memory_dataset = torch.class('torch.memory_dataset')

local initcheck = argcheck{
    pack=true,
    help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up to 14 million+ images)
]],
    {name="split",
        type="number",
        help="Percentage of split to go to Training"},

    {name="samplingMode",
        type="string",
        help="Sampling mode: random | balanced ",
        default = "balanced"},

    {name="classes",
        type="table",
        help="List of class names"},

    {name="samples",
        type="table",
        help="Table containing the data"},

    {name="sampleSize",
        type="table",
        help="a consistent sample size to resize the images"},
}

function memory_dataset:__init(...)

    -- argcheck
    local args = initcheck(...)
    for k,v in pairs(args) do self[k] = v end

    -- find class names
    self.classIndices = {}
    for k,v in ipairs(self.classes) do
        self.classIndices[v] = k
    end

    --==========================================================================
    self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
    self.classList = {}                  -- index of imageList to each image of a particular class
    self.classListSample = self.classList -- the main list used when sampling data

    self.numSamples = self.samples.data:size(1)
    if self.verbose then print(self.numSamples ..  ' samples found.') end

    --==========================================================================
    print('Updating classList and imageClass appropriately')
    self.imageClass:resize(self.numSamples)
    for i=1,#self.classes do
        local idx = self.samples.label:eq(i)
        self.classList[i] = torch.linspace(1, self.numSamples, self.numSamples)[idx]:long()
        self.imageClass[idx] = i
    end

    --==========================================================================
    if self.split == 100 then
        self.testIndicesSize = 0
    else
        print('Splitting training and test sets to a ratio of ' .. self.split .. '/' .. (100-self.split))
        self.classListTrain = {}
        self.classListTest  = {}
        self.classListSample = self.classListTrain
        local totalTestSamples = 0
        -- split the classList into classListTrain and classListTest
        for i=1,#self.classes do
            local list = self.classList[i]
            local count = self.classList[i]:size(1)
            local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
            local perm = torch.randperm(count)
            self.classListTrain[i] = torch.LongTensor(splitidx)
            for j=1,splitidx do
                self.classListTrain[i][j] = list[perm[j]]
            end
            if splitidx == count then -- all samples were allocated to train set
                self.classListTest[i]  = torch.LongTensor()
            else
                self.classListTest[i]  = torch.LongTensor(count-splitidx)
                totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
                local idx = 1
                for j=splitidx+1,count do
                    self.classListTest[i][idx] = list[perm[j]]
                    idx = idx + 1
                end
            end
        end
        -- Now combine classListTest into a single tensor
        self.testIndices = torch.LongTensor(totalTestSamples)
        self.testIndicesSize = totalTestSamples
        local tdata = self.testIndices:data()
        local tidx = 0
        for i=1,#self.classes do
            local list = self.classListTest[i]
            if list:dim() ~= 0 then
                local ldata = list:data()
                for j=0,list:size(1)-1 do
                    tdata[tidx] = ldata[j]
                    tidx = tidx + 1
                end
            end
        end
        -- Now combine classListTrain into a single tensor
        self.trainIndicesSize = self.numSamples - totalTestSamples
        self.trainIndices = torch.LongTensor(self.trainIndicesSize)
        local tdata = self.trainIndices:data()
        local tidx = 0
        for i=1,#self.classes do
            local list = self.classListTrain[i]
            if list:dim() ~= 0 then
                local ldata = list:data()
                for j=0,list:size(1)-1 do
                    tdata[tidx] = ldata[j]
                    tidx = tidx + 1
                end
            end
        end
    end
end

-- size(), size(class)
function memory_dataset:size(class, list)
    list = list or self.classList
    if not class then
        return self.numSamples
    elseif type(class) == 'string' then
        return list[self.classIndices[class]]:size(1)
    elseif type(class) == 'number' then
        -- special case for empty tensors
        if list[class]:dim() == 0 then
            return 0
        end
        return list[class]:size(1)
    end
end

-- size(), size(class)
function memory_dataset:sizeTrain(class)
    if self.split == 0 then
        return 0;
    end
    if class then
        return self:size(class, self.classListTrain)
    else
        return self.numSamples - self.testIndicesSize
    end
end

-- size(), size(class)
function memory_dataset:sizeTest(class)
    if self.split == 100 then
        return 0
    end
    if class then
        return self:size(class, self.classListTest)
    else
        return self.testIndicesSize
    end
end

-- getByClass
function memory_dataset:getByClass(class)
    local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
    return self.samples.data[self.classListSample[class][index]]
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable)
    local data, scalarLabels, labels
    local quantity = #scalarTable
    local samplesPerDraw
    if dataTable[1]:dim() == 3 then samplesPerDraw = 1
    else samplesPerDraw = dataTable[1]:size(1) end
    if quantity == 1 and samplesPerDraw == 1 then
        data = dataTable[1]
        scalarLabels = scalarTable[1]
        labels = torch.LongTensor(#(self.classes)):fill(-1)
        labels[scalarLabels] = 1
    else
        data = torch.Tensor(quantity * samplesPerDraw,
            self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
        scalarLabels = torch.LongTensor(quantity * samplesPerDraw)
        labels = torch.LongTensor(quantity * samplesPerDraw, #(self.classes)):fill(-1)
        for i=1,#dataTable do
            data[{{i, i+samplesPerDraw-1}}]:copy(dataTable[i])
            scalarLabels[{{i, i+samplesPerDraw-1}}]:fill(scalarTable[i])
            labels[{{i, i+samplesPerDraw-1},{scalarTable[i]}}]:fill(1)
        end
    end
    return data, scalarLabels, labels
end

-- sampler, samples from the training set.
function memory_dataset:sample(quantity)
    if self.split == 0 then
        error('No training mode when split is set to 0')
    end
    quantity = quantity or 1
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        local class = torch.random(1, #self.classes)
        local out = self:getByClass(class)
        table.insert(dataTable, out)
        table.insert(scalarTable, class)
    end
    local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
    return data, scalarLabels, labels
end

function memory_dataset:get(i1, i2, indexList)
    local indices, quantity
    if type(i1) == 'number' then
        if type(i2) == 'number' then -- range of indices
            indices = torch.range(i1, i2);
            quantity = i2 - i1 + 1;
        else -- single index
            indices = {i1}; quantity = 1
        end
    elseif type(i1) == 'table' then
        indices = i1; quantity = #i1;         -- table
    elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
        indices = i1; quantity = (#i1)[1];    -- tensor
    else
        error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))
    end
    if indexList == nil then
        indexList = self.testIndices
    end
    assert(quantity > 0)
    -- now that indices has been initialized, get the samples
    local dataTable = {}
    local scalarTable = {}
    for i=1,quantity do
        -- load the sample
        local fetchIndex = indexList[indices[i]]
        local out = self.samples.data[fetchIndex]
        table.insert(dataTable, out)
        table.insert(scalarTable, self.imageClass[fetchIndex])
    end
    local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
    return data, scalarLabels, labels
end

function memory_dataset:test(quantity)
    if self.split == 100 then
        error('No test mode when you are not splitting the data')
    end
    local i = 1
    local n = self.testIndicesSize
    local qty = quantity or 1
    return function ()
        if i <= n then
            local data, scalarLabelss, labels = self:get(i, math.min(i+qty-1, n))
            i = i + qty
            return data, scalarLabelss, labels
        end
    end
end

function memory_dataset:train(quantity)
    if self.split == 0 then
        error('No train mode when all data is dedicated to test')
    end
    local i = 1
    local n = self.trainIndicesSize
    local qty = quantity or 1
    return function ()
        if i+qty-1 <= n then
            local data, scalarLabelss, labels = self:get(i, i+qty-1, self.trainIndices)
            i = i + qty
            return data, scalarLabelss, labels
        end
    end
end

function memory_dataset:shuffle(perm)
    -- shuffles the train set
    if perm == nil then
        perm = torch.randperm(self.trainIndicesSize):long()
    end

    if self.split == 0 then
        error('No train mode when all data is dedicated to test')
    end
    self.trainIndices = self.trainIndices:index(1, perm)
end
