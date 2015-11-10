--
-- Created by IntelliJ IDEA.
-- User: lioruzan
-- Date: 06/10/15
-- Time: 14:08
-- To change this template use File | Settings | File Templates.
--

local Type, parent = torch.class('nn.Type', 'nn.Module')

function Type:__init(outtype)
    parent.__init(self)
    self.outtype = outtype
end

function Type:updateOutput(input)
    self.output = input:type(self.outtype)
    return self.output
end

function Type:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:typeAs(input)
    return self.gradInput
end

