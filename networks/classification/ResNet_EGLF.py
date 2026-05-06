import torch
from torch import nn
import torch.nn.functional as F
from networks.classification.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152

def load_original_net(model_deep=50, pretrained=False, in_channels=3, num_classes=1000, ckpt_path='',
                      need_features=True):
    if model_deep == 18:
        expansion = 1
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [64, 64, 64],
                            'layer2': [128, 32, 32],
                            'layer3': [256, 16, 16],
                            'layer4': [512, 8, 8]}
    elif model_deep == 34:
        expansion = 1
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [64, 64, 64],
                            'layer2': [128, 32, 32],
                            'layer3': [256, 16, 16],
                            'layer4': [512, 8, 8]}
    elif model_deep == 50:
        expansion = 4
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    elif model_deep == 101:
        expansion = 4
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    elif model_deep == 152:
        expansion = 4
        layer_shape_dict = {'stem': [64, 64, 64],
                            'layer1': [256, 64, 64],
                            'layer2': [512, 32, 32],
                            'layer3': [1024, 16, 16],
                            'layer4': [2048, 8, 8]}
    else:
        raise ValueError(f'ResNet model deep: {model_deep} is not supported')

    original_net = eval(
        f'resnet{model_deep}(weights={pretrained}, in_channels={in_channels}, num_classes={num_classes}, ckpt_path="{ckpt_path}", need_features={need_features})')
    layer_shape_dict['head'] = [num_classes]
    return original_net, expansion, layer_shape_dict


def calc_edl_metrics(evidence, num_classes):
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    uncertainty = num_classes / S
    probs = alpha / S
    return probs, uncertainty

class ResNet_EGLF(nn.Module):
    def __init__(self, model_deep=50, pretrained=False, in_channels=3, num_classes=1000, ckpt_path='',
                 need_layer_name_list=None, num_patches=5):
        super(ResNet_EGLF, self).__init__()

        if need_layer_name_list is None:
            need_layer_name_list = ['layer1', 'layer2', 'layer3', 'layer4']
        self.need_layer_name_list = need_layer_name_list
        print(f'need_layer_name_list: {self.need_layer_name_list}')

        self.num_classes = num_classes
        self.num_patches = num_patches

        self.original_net, self.expansion, self.layer_shape_dict = load_original_net(model_deep,
                                                                                     pretrained,
                                                                                     in_channels=in_channels,
                                                                                     num_classes=num_classes,
                                                                                     ckpt_path=ckpt_path,
                                                                                     need_features=True)

        # for name, child in self.original_net.named_children():
        #     if name in ['stem', 'layer1', 'layer2', 'layer3']:
        #         for param in child.parameters():
        #             param.requires_grad = True
        #     else:
        #         for param in child.parameters():
        #             param.requires_grad = True
        for p in self.original_net.parameters():  # Freezing backbone
            p.requires_grad = True


        need_dim_list = []

        axis_reduction = 16
        layer_name = self.need_layer_name_list[-1]
        if 'head' in self.need_layer_name_list:
            need_dim_list.append(num_classes)
            self.head_relu = nn.ReLU(inplace=True)

        num_scale = len(need_dim_list)
        EN_dim_list = []
        for first_dim in need_dim_list:
            EN_dim_list.append([first_dim])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.edl_classifier_global = nn.Sequential(
            nn.Linear(self.layer_shape_dict[layer_name][0], num_classes),
            nn.Softplus()
        )
        self.edl_classifier_local = nn.Sequential(
            nn.Linear(self.layer_shape_dict[layer_name][0], num_classes),
            nn.Softplus()
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(self.layer_shape_dict[layer_name][0]*2, num_classes)
        )
        

    def forward(self, x):
        out, feature_list = self.original_net(x)

        cls_features = []
        for layer_name in self.need_layer_name_list:
            if layer_name == 'head':
                continue
            cls_features.append(feature_list[layer_name])

        logit = cls_features[len(cls_features)-1]
        P, C, H, W = logit.shape
        view_logit = logit.view(-1, self.num_patches+1, C, H, W)
        
        global_features = view_logit[:, 0, :, :, :]
        local_features = view_logit[:, 1:, :, :, :]

        global_feat_flat = self.avgpool(global_features).flatten(1)
        
        global_evidence = self.edl_classifier_global(global_feat_flat)
        
        global_alpha = global_evidence + 1
        global_probs = global_alpha / torch.sum(global_alpha, dim=1, keepdim=True)

        local_logit = self.avgpool(local_features.flatten(0, 1))
        local_logit = torch.flatten(local_logit, 1)
        
        local_evidence = self.edl_classifier_local(local_logit)
        local_probs_all, uncertainty = calc_edl_metrics(local_evidence, self.num_classes)
        
        local_probs_all = local_probs_all.view(-1, self.num_patches, self.num_classes)
        uncertainty = uncertainty.view(-1, self.num_patches, 1)
        local_evidence = local_evidence.view(-1, self.num_patches, self.num_classes)

        max_probs, _ = torch.max(local_probs_all, dim=2, keepdim=True)
        scores = max_probs * (1 - uncertainty)
        scores = scores.squeeze(-1)
        
        attention_weights = torch.nn.functional.softmax(scores, dim=1)
    
        aggregated_local_probs = (attention_weights.unsqueeze(-1) * local_probs_all).sum(dim=1)

        local_feats_vec = local_logit.view(-1, self.num_patches, C)
        weights_for_bmm = attention_weights.unsqueeze(1)
        
        weighted_local_feat = torch.bmm(weights_for_bmm, local_feats_vec).squeeze(1)
        
        fused_feat = torch.cat([global_feat_flat, weighted_local_feat], dim=1)
        
        out = self.final_classifier(fused_feat)

        return out, local_evidence, global_evidence, global_probs, aggregated_local_probs