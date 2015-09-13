require 'nn'
require 'cunn'
require 'cudnn'
require 'image'

-- Same as scratch training directly with pruning.
print '==> building model'

local model = nn.Sequential()

do -- data augmentation module
    local Blur,parent = torch.class('nn.Blur', 'nn.Module')

    local function gaussian(n, sigma)
        local n_hat = torch.floor(n/2)
        local X = torch.range(-n_hat,n_hat):repeatTensor(n,1)
        local Y = X:t()

        local g = torch.exp(-(torch.pow(X,2) + torch.pow(Y,2))/(2*sigma*sigma))
        return g/g:sum()
    end

    function Blur:__init(n, sigma)
        parent.__init(self)
        self:updateGaussian(n, sigma)
    end

    function Blur:updateGaussian(n, sigma)
        self.g = gaussian(n, sigma):repeatTensor(3,1,1)
    end

    function Blur:updateOutput(input)
        if input:dim() == 3 then
            self.output = image.convolve(input,self.g,'same')
        else
            if (self.output == nil) or (self.output:dim() ~= 3) then
                self.output = torch.Tensor():resizeAs(input):typeAs(input)
            end

            for i = 1,input:size(1) do
                self.output[i] = image.convolve(input[i],self.g,'same')
            end
            self.output = self.output[{{1,input:size(1)}}]
        end

        return self.output
    end
end

-- Conv11 & Conv12 layers
model:add(nn.Blur(1,1))
model:add(cudnn.SpatialConvolution(1, 32, 3, 3, 1, 1, 1, 1)) --1
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) --3
model:add(cudnn.ReLU(true))

-- Pool1
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv21 & Conv22 layers
model:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1)) -- 6
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1)) -- 8
model:add(cudnn.ReLU(true))

-- Pool2
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv31 & Conv 32 layers
model:add(cudnn.SpatialConvolution(128, 96, 3, 3, 1, 1, 1, 1)) -- 11
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(96, 192, 3, 3, 1, 1, 1, 1)) -- 13
model:add(cudnn.ReLU(true))

-- Pool3
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv41 & Conv42
model:add(cudnn.SpatialConvolution(192, 128, 3, 3, 1, 1, 1, 1)) -- 16
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1)) -- 18
model:add(cudnn.ReLU(true))

-- Pool 4
model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

-- Conv51 & Conv52
model:add(cudnn.SpatialConvolution(256, 160, 3, 3, 1, 1, 1, 1)) -- 21
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialConvolution(160, 320, 3, 3, 1, 1, 1, 1)) -- 23

-- Pool 5
model:add(cudnn.SpatialAveragePooling(6, 6, 1, 1))

-- Dropout
model:add(nn.Dropout(0.4))

-- Fc6
model:add(nn.Reshape(320,true))
model:add(nn.Linear(320, #dataset.classes))
model:add(nn.LogSoftMax())

return model
