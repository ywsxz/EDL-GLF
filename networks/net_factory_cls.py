import torch
from torchvision import models

from networks.classification.ResNet import *

# target net
from networks.classification.ResNet_EGLF import ResNet_EGLF

def net_factory(args):
    net_type = args.model
    net_deep = args.model_deep
    in_channels = args.in_channels
    num_classes = args.num_classes
    ckpt_path = args.ckpt_path
    need_layer_name_list = args.need_layer_name_list
    num_patches = args.num_patches

    if (args.pretrained == 1) or (args.pretrained == '1'):
        pretrained = True
    else:
        pretrained = False

    if net_type == 'ResNet':
        if net_deep in [18, 32, 50, 101, 152]:
            net = eval(
                f'resnet{net_deep}(weights={pretrained}, in_channels={in_channels}, num_classes={num_classes}, ckpt_path="{ckpt_path}")')
        else:
            raise ValueError(f'ResNet not support deep [18, 32, 50, 101, 152]: {net_deep}')
        
    elif net_type == 'ResNet_EGLF':
        net = ResNet_EGLF(model_deep=net_deep, pretrained=True, in_channels=in_channels,
                         num_classes=num_classes, ckpt_path=ckpt_path,
                         need_layer_name_list=need_layer_name_list, num_patches=num_patches)
        
    
    

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    return net