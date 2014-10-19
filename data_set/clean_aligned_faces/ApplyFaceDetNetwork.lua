require 'image'
require 'nn'

criterion = 'NLL'

-- preparations
dataDir = 'data'
inputFilePath = paths.concat(dataDir, 'images_unknown.txt')
outputFileName = paths.concat(dataDir, 'images_unknown_NLL.txt')
model = torch.load('./train-a-face-detector_NLL/face.net')

-- apply network on iterated files
fid = io.open(outputFileName, "w")
for imPath in io.lines(inputFilePath) do
    k = string.find(imPath, ',')
    if k then
       imPath = string.sub(imPath, 1, k-1)
    end
    print(imPath)
    im = image.load(imPath, 1)
    im:resize(1, 152, 152)
    prediction = model:forward(im)
    prediction_1d = torch.FloatTensor(2)
    prediction_1d:copy(prediction)
    if (criterion == 'NLL') then
	--- the last layer is LogSoftMax
        prediction_1d[1] = math.exp(prediction_1d[1])
        prediction_1d[2] = math.exp(prediction_1d[2])
    end
    fid:write(imPath,',',tostring(prediction_1d[1]),',',tostring(prediction_1d[2]),'\n')
end
fid:close()
