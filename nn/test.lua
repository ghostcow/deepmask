----------------------------------------------------------------------
-- Deepface nn test code torch7
-- testing on the test data from cfw dataset (later we will test on LFW)
-- Uses CUDA
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- turn off dropout modules
   for iModule = 1,#model.modules do
      if (torch.type(model.modules[iModule]) == 'nn.Dropout') then
         model.modules[iModule].train = false
      end
   end

   -- local vars
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over test data
   print('==> testing on test set:')
   local totalSize = 0
   local testDataChunk
   for iChunk = 1,testData.numChunks do
   	testDataChunk = testData.getChunk(iChunk)
  	totalSize = totalSize + testDataChunk:size()
	for t = 1,testDataChunk:size(),opt.batchSize do
		-- disp progress
		xlua.progress(t, testDataChunk:size())
		if ((t+opt.batchSize-1) > testDataChunk:size()) then
			-- we don't use the last samples
			break
		end

		-- get new sample
		local inputs = testDataChunk.data[{{t,t+opt.batchSize-1}}]
        local numInputs = inputs:size()[1]
		inputs = inputs:cuda()
		local targets = testDataChunk.labels[{{t,t+opt.batchSize-1}}]

		-- test sample
		local outputs = model:forward(inputs)
		for i=1,numInputs do
			confusion:add(outputs[i], targets[i])
		end

        -- grabage collection after every batch (TODO : might be too expensive...)
        collectgarbage()
	end
   end

   -- timing
   time = sys.clock() - time
   time = time / totalSize
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix values
   confusion:updateValids()
   print("accuracy = ", confusion.totalValid * 100)
   local filename_confusion = paths.concat(opt.save, 'confusion_test')
   torch.save(filename_confusion, confusion)
   testLogger:add{['% total accuracy'] = confusion.totalValid * 100, 
     ['% average accuracy'] = confusion.averageValid * 100}
   if opt.plot then
      testLogger:style{['% total accuracy'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   
   -- next iteration:
   confusion:zero()

   -- turn on dropout modules
   for iModule = 1,#model.modules do
      if (torch.type(model.modules[iModule]) == 'nn.Dropout') then
         model.modules[iModule].train = true
      end
   end
end
