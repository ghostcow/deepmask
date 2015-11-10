require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'paths'

print(cudnn.benchmark)
print(cudnn.fastest)
cudnn.benchmark = true
cudnn.fastest = true

-- 1. Create Network
if opt.retrain ~= 'none' then
    assert(false, 'Retrain is not implemented')
else
    local networkConfigPath = 'models/' .. opt.netType .. '.lua'
    print('=> Creating model from file: ' .. networkConfigPath)
    mask, score = require(networkConfigPath)
--    -- Initilize model according to MSR
--    print('=> Initializing weights according to MSR')
--    local function MSRinit(net)
--        local function init(name)
--            for _,v in pairs(net:findModules(name)) do
--                local n = v.kW*v.kH*v.nInputPlane
--                v.weight:normal(0,math.sqrt(2/n))
--                v.bias:zero()
--            end
--        end
--        init'cudnn.SpatialConvolution'
--        init'nn.SpatialConvolution'
--    end
--    MSRinit(model)
end

-- 2. Create Criterion
--TODO: define criterions
criterion = nn.ClassNLLCriterion()
print('=> Model')
print(model)
print('=> Criterion')
print(criterion)

score:cuda()
mask:cuda()
--
---- 3. Check for parallel training mode
--if opt.parallel then
--    net = nn.DataParallelTable(1)
--    for i = opt.gpu, (opt.gpu+1) do
--        cutorch.setDevice(i)
--        net:add(model:clone():cuda(), i)
--    end
--
--    cutorch.setDevice(opt.gpu)
--else
--    model:cuda()
--    net = model
--end

-- 4. Convert Criterion to CUDA
print('=> Converting Criterion to CUDA')
criterion:cuda()
