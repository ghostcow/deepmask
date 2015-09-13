require 'nn'

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