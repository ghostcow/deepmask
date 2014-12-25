require 'image'
require 'deep_id2_utils'

im1 = image.load('../../project_report/test_aligned.jpg')
im2 = image.load('../../project_report/test2_aligned.jpg')

im1 = image.scale(im1, 115, 140)
im2 = image.scale(im2, 115, 140)

images = torch.Tensor(2, 3, 140, 115)
images[1] = im1
images[2] = im2

for iPatch = 1,25 do
    patches = DeepId2Utils.getPatch(images, iPatch, true)
    image.save('deepId2/im1_patch_'..iPatch..'.png', patches[1])
    image.save('deepId2/im2_patch_'..iPatch..'.png', patches[2])

    image.save('deepId2/im1_patch_'..iPatch..'_flipped.png', patches[3])
    image.save('deepId2/im2_patch_'..iPatch..'_flipped.png', patches[4])
end