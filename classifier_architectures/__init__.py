from classifier_architectures.vgg import VGG_net as VGG
from classifier_architectures.inception import GoogLeNet
from classifier_architectures.resnet import ResNet50, ResNet101, ResNet152
from classifier_architectures.efficientnet import EfficientNet
from classifier_architectures.custom import M as FeedForwardClassifier
from classifier_architectures.custom import ClassifierModel as ShortConv
from classifier_architectures.custom import ConvNet as TallConv