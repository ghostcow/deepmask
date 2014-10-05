require 'ccn2'

local ccntest_jac = {}
local errs = {}
precision = 1e-3

function ccntest_jac.SpatialConvolution_Jacobian()
    local bs = 128
    local from = 32
    local to = 16
    local ki = 9
    local si = 1
    local outi = 9 --63
    local ini = (outi-1)*si+ki

    local tm = {}
    local title = string.format('ccn2.SpatialConvolution %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
                                bs, from, ini, ini, ki, ki, bs, to, outi, outi, si, si)
    local layer_notation = string.format('%dx%dx%dx%d@%dx%d', to, ki, ki, from, outi, outi)    
    errs['ccn2.SpatialConvolution - '..layer_notation] = tm

    local module = ccn2.SpatialConvolution(from,to,ki,si):cuda()
    local input = torch.randn(from,ini,ini,bs):cuda():zero()

    local err = jac.testJacobian(module,input)
    print(err)
    mytester:assertlt(err,precision, 'error on state ')
    tm.err = err
end

function ccntest_jac.SpatialConvolutionLocal_Jacobian()
    local bs = 32
    local from = 16
    local to = 16
    local ki = 9
    local si = 1 
    local outi = 9 -- 55
    local ini = (outi-1)*si+ki

    local tm = {}
    local title = string.format('ccn2.SpatialConvolutionLocal %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d]',
                                bs, from, ini, ini, ki, ki, bs, to, outi, outi, si, si)
    local layer_notation = string.format('%dx%dx%dx%d@%dx%d', to, ki, ki, from, outi, outi)    
    errs['ccn2.SpatialConvolutionLocal - '..layer_notation] = tm    

    local module = ccn2.SpatialConvolutionLocal(from,to,ini,ki,si):cuda()
    local input = torch.randn(from,ini,ini,bs):cuda():zero()

    local err = jac.testJacobian(module,input)
    mytester:assertlt(err,precision, 'error on state ')
    tm.err = err
end

math.randomseed(os.time())
jac = ccn2.Jacobian
mytester = torch.Tester()
mytester:add(ccntest_jac)
mytester:run(tests)
print ''
print ' ---------------------------------------------------------------------------------------------------'
print '|  Module                                                                          |  Jacobian err |'
print ' ---------------------------------------------------------------------------------------------------'
for module,tm in pairs(errs) do
    local str = string.format('| %-80s | %4.5f       |', module, tm.err)
    print(str)
end
print ' ---------------------------------------------------------------------------------------------------'
