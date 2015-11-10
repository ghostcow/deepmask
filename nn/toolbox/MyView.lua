local MyView, parent = torch.class('nn.MyView', 'nn.Module')

function MyView:__init(h,w)
    parent.__init(self)
    self.h = h
    self.w = w
    self.gradInput = nil
    self.output = nil
end

function MyView:updateOutput(input)
    local isz = input:size()
    local dim = isz:size()
    local osz = torch.LongStorage(dim + 1)
    --print(osz:size(), isz:size(), dim)
    for i=1,dim do
        osz[i] = isz[i]
    end
    osz[dim] = self.h
    osz[dim+1] = self.w
    if isz[dim] ~= self.h * self.w then
        error(string.format(
            'input view (%s) and desired view (%s) do not match',
            table.concat(input:size():totable(), 'x'),
            table.concat(osz:totable(), 'x')))
    end
    self.output = input:view(osz)
    return self.output
end

function MyView:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput:view(input:size())
    return self.gradInput
end
