require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'paths'

--[[
1. Create Model
2. Create Criterion
3. If preloading option is set, preload weights from existing models appropriately
4. Convert model to CUDA
]]--

-- parse command line arguments
if not opt then
    print '==> processing options'
    opt = getOptions()
end

-- 1. Create Network
if opt.retrain ~= 'none' then
    assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
    print('Loading model from file: ' .. opt.retrain);
    model = torch.load(opt.retrain)
else
    local networkConfigPath = 'models/' .. opt.netType .. '.lua'
    print('=> Creating model from file: ' .. networkConfigPath)
    model = require(networkConfigPath)
end

-- 2. Initilize model according to MSR
print('=> Initializing weights according to MSR')
local function MSRinit(net)
    local function init(name)
        for _,v in pairs(net:findModules(name)) do
            local n = v.kW*v.kH*v.nOutputPlane
            v.weight:normal(0,math.sqrt(2/(1.25*n))) -- 1.25 is the initial value
            v.bias:zero()
        end
    end
    init'nn.SpatialConvolution'
end
MSRinit(model)

-- 3. Create Criterion
criterion = nn.ClassNLLCriterion()
print('=> Model')
print(model)
print('=> Criterion')
print(criterion)

-- 4. Convert model to CUDA
print('=> Converting model to CUDA')
model:cuda()
criterion:cuda()
