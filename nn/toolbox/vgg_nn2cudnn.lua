require 'nn'
require 'cudnn'
torch.setdefaulttensortype('torch.FloatTensor')

local cudnn_vgg = nn.Sequential()
cudnn_vgg:add(cudnn.SpatialConvolution(3, 64, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(64, 64, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialMaxPooling(2,2,2,2))
cudnn_vgg:add(cudnn.SpatialConvolution(64, 128, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(128, 128, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialMaxPooling(2,2,2,2))
cudnn_vgg:add(cudnn.SpatialConvolution(128, 256, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(256, 256, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialMaxPooling(2,2,2,2))
cudnn_vgg:add(cudnn.SpatialConvolution(256, 512, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialMaxPooling(2,2,2,2))
cudnn_vgg:add(cudnn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))
cudnn_vgg:add(cudnn.ReLU(true))
cudnn_vgg:add(cudnn.SpatialConvolution(512, 512, 3,3, 1,1, 1,1))

local vgg = torch.load('/home/lioruzan/obj_detection_proj/deepmask_nn/nn/vgg_model_d_e/vgg16_deepmask.t7')
cudnn_vgg:share(vgg, 'weight', 'bias')

for i=1,#cudnn_vgg.modules do
    local layer = cudnn_vgg:get(i)
    local layer2 = vgg:get(i)
    if layer.weight then
        if layer.weight.data() ~= layer2.weight.data() then print('error in layer '..tostring(i)) end
        if layer.bias.data() ~= layer2.bias.data() then print('error in layer '..tostring(i)) end
    end
end

torch.save('/home/lioruzan/obj_detection_proj/deepmask_nn/nn/vgg_model_d_e/cudnn_vgg16_deepmask.t7', cudnn_vgg)