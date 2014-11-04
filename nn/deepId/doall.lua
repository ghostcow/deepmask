package.path = package.path .. ";../?.lua"
require 'options'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()
opt.save = paths.concat('../../results_deepid/', opt.save)
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
--------- write options table to log file ----------------------------
os.execute('mkdir -p ' .. opt.save)
logFilePath = paths.concat(opt.save, 'doall.log')
fid = io.open(logFilePath,"a")
fid:write(os.date("%Y_%m_%d_%X"), '\n')
fid:write("opt : {")
for filedName,fieldValue in pairs(opt) do
    fid:write(filedName, " = ", tostring(fieldValue), ', ')
end
fid:write("}\n\n")
fid:close()

----------------------------------------------------------------------
print '==> executing all'
patchIndex = opt.patchIndex
dofile 'data_patch.lua'

k = string.find(opt.modelName, '.', 1, true)
if k then
    modelFileName = opt.modelName:sub(1, k-1)
end

dofile('../'..modelFileName..'.lua')
dofile('../train.lua')
if not opt.trainOnly then
    dofile('../test.lua')
end

----------------------------------------------------------------------
print '==> training!'

while true do
    train()
    if not opt.trainOnly then
        test()
    end
end