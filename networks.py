# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
from tensorflow import keras
from tensorflow.keras import layers

kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, name='resnet'):
    x = layers.ZeroPadding2D(padding=3, name=f'{name}.conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(x)
    x = layers.ReLU(name=f'{name}.relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name=f'{name}.maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name=f'{name}.maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name=f'{name}.layer1')
    x = make_layer(x, 128, blocks_per_layer[1], name=f'{name}.layer2')
    x = make_layer(x, 256, blocks_per_layer[2], name=f'{name}.layer3')
    x = make_layer(x, 512, blocks_per_layer[3], name=f'{name}.layer4')

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def fcn(x, num_output_channels=1):
    x = resnet18(x, name='resnet18')
    x = layers.Conv2D(filters=128, kernel_size=1, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.UpSampling2D(interpolation='bilinear', name='up1')(x)
    x = layers.Conv2D(filters=32, kernel_size=1, name='conv2')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.UpSampling2D(interpolation='bilinear', name='up2')(x)
    return layers.Conv2D(filters=num_output_channels, kernel_size=1, name='conv3')(x)
