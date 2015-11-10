local InterleaveTable, parent = torch.class('nn.InterleaveTable', 'nn.Module')

function InterleaveTable:__init()
  parent.__init(self)
  self.output = nil
  self.gradInput = nil
end

-- interleave or de-interleave table of torch.Tensors of the same size to produce single output
local function interleave(inputs, output, separate)
  local n = #inputs
  for i = 1,n do
    for j = 1,inputs[i]:nElement() do
      if separate then
        inputs[i]:storage()[j] = output:storage()[i + n * (j-1)]
      else
        output:storage()[i + n * (j-1)] = inputs[i]:storage()[j]
      end
    end
  end
end

-- interleave input table -> output in last two dimensions
-- #input must be a square
-- all input tensors must be contiguous
function InterleaveTable:updateOutput(input)
  local isz = input[1]:size()
  -- assert input is legal
  for k,v in pairs(input) do
    local vsz = v:size()
    for i=1,vsz:size() do assert(isz[i] == vsz[i]) end
    assert(input[1]:type() == v:type())
    assert(v:isContiguous())
  end
  -- calc output size
  local osz = isz.new():resize(isz:size()):copy(isz)
  local fact = torch.sqrt(#input)
  osz[#osz-1] = fact * osz[#osz-1]
  osz[#osz] = fact * osz[#osz]
  self.output = input[1].new():resize(osz)
  -- interleave inputs
  interleave(input, self.output, false)
  return self.output
end

function InterleaveTable:updateGradInput(input, gradOutput)
  self.gradInput={}
  for k,v in ipairs(input) do
    table.insert(self.gradInput, v.new():resizeAs(v))
  end
  -- separate from gradOutput to gradInput table
  interleave(self.gradInput, gradOutput, true)
  return self.gradInput
end

