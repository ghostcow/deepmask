----------------------------------------------------------------------
-- test procedure definiton, assumes dataset is defined
----------------------------------------------------------------------

require 'os'
require 'torch'
require 'cutorch'
require 'xlua'
require 'optim'
require 'logger'

----------------------------------------------------------------------
print '==> defining some tools for test'
-- This matrix records the current confusion across classes
if confusion == nil then
    confusion = optim.ConfusionMatrix(dataset.classes)
end

----------------------------------------------------------------------
print '==> defining test procedure'

function test()
   -- set all modules to test
   net:evaluate()

   -- local vars
   local timer = torch.Timer()
   local t = 0

   -- test over test data
   print('==> testing on test set:')
   for inputs, labels in dataset:test(opt.batchSize) do
       -- disp progress
       xlua.progress(t, dataset:sizeTest())
       t = t + inputs:size(1)

       inputs = inputs:cuda()
       labels = labels:double():cuda()

       -- test sample
       local outputs = net:forward(inputs)

       for i=1,inputs:size()[1] do
           confusion:add(outputs[i], labels[i])
       end

       -- grabage collection after every batch
       collectgarbage()
   end

   -- timing
   local time = timer:time().real / dataset:sizeTest()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix values
   confusion:updateValids()
   logTest(confusion, nil, net)

   -- next iteration:
   confusion:zero()
end
