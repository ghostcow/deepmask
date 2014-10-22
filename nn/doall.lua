require 'options'

----------------------------------------------------------------------
print '==> processing options'
opt = getOptions()

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
if opt.useDatasetChunks then
    print '==> datasets is read in chunks'
    dofile 'metadata.lua'
else
    dofile 'data.lua'
end

dofile(opt.modelName..'.lua')
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

while true do
    train()
    test()
end
