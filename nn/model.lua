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
    mask, score = dofile(networkConfigPath)
    if mask == nil then print('mask is nil') end
    if score == nil then print('score is nil') end
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
-- mask criterion
maskCriterion = nn.BCECriterion()
print('=> Mask Prediction Model')
print(mask)
print('=> Mask Criterion')
print(maskCriterion)
-- score criterion
scoreCriterion = nn.CrossEntropyCriterion()
print('=> Score Prediction Model')
print(score)
print('=> Score Criterion')
print(scoreCriterion)

-- 4. Convert Criterion to CUDA
print('=> Converting Criterions and Models to CUDA')
mask:cuda()
maskCriterion:cuda()
score:cuda()
scoreCriterion:cuda()
