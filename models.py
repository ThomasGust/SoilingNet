from keras_segmentation.models.fcn import fcn_32
from keras_segmentation.models.unet import resnet50_unet
from keras_segmentation.models.segnet import resnet50_segnet


fcn32 = fcn_32(8, input_height=224, input_width=224)
resnetunet = resnet50_unet(8, 224, 224)
resnetsegnet = resnet50_segnet(8, 224, 224)