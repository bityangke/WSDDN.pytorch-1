'''
Function:
    vgg
Author:
    Charles
'''
import torch
import torchvision


'''vgg from torchvision==0.4.0'''
def VGGs(vgg_type, pretrained=False):
    if vgg_type == 'vgg11':
        model = torchvision.models.vgg11(pretrained=pretrained)
    elif vgg_type == 'vgg11_bn':
        model = torchvision.models.vgg11_bn(pretrained=pretrained)
    elif vgg_type == 'vgg13':
        model = torchvision.models.vgg13(pretrained=pretrained)
    elif vgg_type == 'vgg13_bn':
        model = torchvision.models.vgg13_bn(pretrained=pretrained)
    elif vgg_type == 'vgg16':
        model = torchvision.models.vgg16(pretrained=pretrained)
    elif vgg_type == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=pretrained)
    elif vgg_type == 'vgg19':
        model = torchvision.models.vgg19(pretrained=pretrained)
    elif vgg_type == 'vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained=pretrained)
    else:
        raise ValueError('Unsupport vgg_type <%s>...' % vgg_type)
    return model