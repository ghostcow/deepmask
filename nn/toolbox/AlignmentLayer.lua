local AlignmentLayer, parent = torch.class('nn.AlignmentLayer', 'nn.Module')

-- pad input height\width dims to force (size // fact) % ils == ils - 1,
-- for interleave trick to work
function AlignmentLayer:__init(div_factor, interleave_size)
    parent.__init(self)
    self.fact = div_factor
    self.ils = interleave_size    
    self.zeropad = nn.SpatialZeroPadding(0)
    self.output = self.zeropad.output
    self.gradInput = self.zeropad.gradInput
end

local function getZeroPaddingParams(size, fact, ils)
    --[[ get zero padding lengths for a certain dimension.
    parameters:
        size: length of input dimension
        fact: division factor at net output before trick
        ils: interleave size- upsampling factor achieved by interleave trick
    ]]
    local pada, padb
    if torch.floor(size/fact) % (ils) == 1 then
        pada, padb = 0,0
    else
        local pad = fact * ( (ils-1) - (torch.floor(size/fact)%ils) ) - size%fact
        if pad % 2 == 0 then
            pada, padb = pad/2, pad/2
        else
            pada = torch.floor(pad/2) + torch.random(0,1)
            padb = pad - pada
        end
    end
    return pada, padb
end

function AlignmentLayer:updateOutput(input)
    local height, width
    -- input must have 3 (CHW) or 4 channels (BCHW). 
    -- B batchsize, C channels, H height, W width
    if input:dim() == 3 then
        height = input:size(2)
        width = input:size(3)
    elseif input:dim() == 4 then
        height = input:size(3)
        width = input:size(4)     
    else
        print('wtf')
    end
    local szp = self.zeropad
    szp.pad_l, szp.pad_r = getZeroPaddingParams(width, self.fact, self.ils)
    szp.pad_t, szp.pad_b = getZeroPaddingParams(height, self.fact, self.ils)
    -- self.zeropad = nn.SpatialZeroPadding(left_pad, right_pad, up_pad, down_pad)
    self.output = self.zeropad:updateOutput(input)
    return self.output
end

-- only call after a forward() with same input
function AlignmentLayer:updateGradInput(input, gradOutput)
--     print(gradOutput:size())
    self.gradInput = self.zeropad:updateGradInput(input, gradOutput)
    return self.gradInput
end
