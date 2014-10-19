require 'optim'
require 'nn'
require 'nnx'
require 'DataSet'

dataset = '/media/data/datasets/CFW/clean_aligned_faces_network/'
patches = 1600
testRatio = 0.2
confusion = optim.ConfusionMatrix { 'Face', 'Background' }
criterion = 'NLL' -- MSE / NLL

-- load model :
model = torch.load('./train-a-face-detector_NLL/face.net')

-- dataset :
dataFace = DataSet {
    dataSetFolder = paths.concat(dataset, 'true'),
    cacheFile = paths.concat(dataset, 'true'),
    nbSamplesRequired = patches,
    channels = 1
}
dataFace:shuffle()
dataBG = DataSet {
    dataSetFolder = paths.concat(dataset, 'false'),
    cacheFile = paths.concat(dataset, 'false'),
    nbSamplesRequired = patches,
    channels = 1
}
dataBG:shuffle()

-- pop subset for testing
testFace = dataFace:popSubset { ratio = testRatio }
testBg = dataBG:popSubset { ratio = testRatio }
testData = nn.DataList()
testData:appendDataSet(testFace, 'Faces')
testData:appendDataSet(testBg, 'Background')
print('------------ Test Data ------------')
print(testData)
if (criterion == 'NLL') then
    testData.targetIsProbability = true
end

function test(dataset)
    -- local vars
    local time = sys.clock()

    -- test over given dataset
    print('<trainer> on testing Set:')
    for t = 1, dataset:size() do
        -- disp progress
        -- xlua.progress(t, dataset:size())

        -- get new sample
        local sample = dataset[t]
        local input = sample[1]
        local expectedOutput = sample[2]
        if (criterion == 'NLL') then
            _, expectedOutput = expectedOutput:max(1)
            expectedOutput = expectedOutput[1]
        end

        local output = model:forward(input)

        -- test sample
        confusion:add(output, expectedOutput)
    end

    -- timing
    time = sys.clock() - time
    time = time / dataset:size()
    print("<trainer> time to test 1 sample = " .. (time * 1000) .. 'ms')

    -- print confusion matrix
    print(confusion)
    confusion:zero()
end

test(testData)
