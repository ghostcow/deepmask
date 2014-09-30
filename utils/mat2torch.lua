require 'mattorch'
matFilePath = arg[1]

x = mattorch.load(matFilePath)
torchFilePath = string.sub(matFilePath, 1, -4)..'t7'
-- torch.save(torchFilePath, x)