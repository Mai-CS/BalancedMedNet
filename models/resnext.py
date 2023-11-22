import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls
import math


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return self.s * out

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if(pretrained):
        if(arch == 'resnet50'):
            print("Loading pretrained weights from: " + model_urls['resnet50'] + "...")
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        elif(arch == 'resnext50'):
            print("Loading pretrained weights from: " + model_urls['resnet50'] + "...")
            model.load_state_dict(model_zoo.load_url(model_urls['resnext50']), strict=False)

    
    return model

def resnet50(num_classes = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained,
                   **kwargs)

def resnext50(num_classes = None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained ,**kwargs)

def crossformer(num_classes = None, out_dim = 2048):
    """
    `CrossFormer(num_classes = num_classes, dim = (64, 128, 256, 512), depth = (2, 2, 8, 2),
    global_window_size = (8, 4, 2, 1), local_window_size = 7)`
    
    The `dim` parameter is a tuple of the number of channels in each of the four stages of the
    CrossFormer. The `depth` parameter is a tuple of the number of CrossFormer blocks in each of the
    four stages. The `global_window_size` parameter is a tuple of the global window size in each of the
    four stages. The `local_window_size` parameter is the local window size
    
    :param num_classes: number of classes to predict
    :return: A CrossFormer model with the specified parameters.
    """
    from models.crossformer import CrossFormer

    model = CrossFormer(
        num_classes = num_classes,        
        dim = (64, 128, 256, 512), # should be comparable in size to resnet50      
        depth = (2, 2, 8, 2),             
        global_window_size = (8, 4, 2, 1), 
        local_window_size = 7, 
        out_dim =  out_dim         
    )
    return model

def vit_small(num_classes = None, out_dim = 2048):
    """
    > The function `vit_small` returns a ViT model with a patch size of 16, a depth of 6, and a hidden
    dimension of 1024
    
    :param num_classes: number of classes in the dataset
    :param out_dim: the dimension of the output of the model, defaults to 2048 (optional)
    :return: A model
    """
    from models.vit_for_small_dataset import ViT

    model = ViT(
        image_size = 224,
        patch_size = 16,
        num_classes = num_classes,
        dim = 2048,
        depth = 6,
        heads = 8,
        mlp_dim = 128,
        dropout = 0.1,
        emb_dropout = 0.1
    )        
        
    return model

model_dict = {
    'resnet50': [resnet50, 2048],
    'resnext50': [resnext50, 2048],
    'crossformer': [crossformer, 2048],
    'vit_small': [vit_small, 2048]
}

class PrototypeRecalibrator():
    def __init__(self, beta, initial_wc, num_classes, cls_num_list, static = False, tau = 1.0):
        self.beta = beta # smoothing coefficient
        self.wc = [initial_wc for _ in range(num_classes)]
        self.num_classes = num_classes
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.static = static
    
    def update(self, prototypes, features, targets):
        # update based on a batch of data
        # use an exponential moving average
        # print("targets: ",targets)
        # print(self.num_classes)
        # print(prototypes.shape)
        bs = int(features.shape[0] / 3)
        f1, _, _ = torch.split(features, [bs, bs, bs], dim=0)
        for i in range(self.num_classes):
            indices = [j for j, x in enumerate(targets.tolist()) if x == i]
            N = len(indices)
            features_i = f1[indices]
            if(N == 0):
                continue
            N = (1 / N)
            exps = torch.zeros(features_i.shape, dtype=torch.float64)
            for j in range(len(features_i)):
                exps[:,j] = 1 / (1 + torch.exp(-1 * torch.dot(features_i[j].T, prototypes[i])))
            wc_batch = N * torch.sum(exps)
            self.wc[i] = (self.beta * self.wc[i] + (1 - self.beta) * wc_batch).item()
    
    def recalibrate(self, prototypes):
        new_prototypes = prototypes.clone()
        # recalibrate prototypes
        if(self.static):
            return new_prototypes + self.m_list.reshape(self.num_classes, -1)
        for i in range(self.num_classes):
            new_prototypes[i] = prototypes[i] + math.log(self.wc[i])
        return new_prototypes
    

class PrototypeStore():
    def __init__(self, num_classes, feat_dim, device, momentum = 0.97, queue_size = 300):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.momentum = momentum
        self.prototypes = torch.zeros((num_classes, feat_dim), dtype=torch.float32, device=device)
        self.queue = [[] for _ in range(num_classes)]
        self.queue_size = queue_size
    
    @torch.no_grad()
    def update_prototypes(self, features, targets):
        '''
        Updates the centroids of the clusters
        '''
        batch_size = int(targets.shape[0])
        features, _ , _ = torch.split(features, [batch_size, batch_size, batch_size], dim=0)

        # update the queue
        for clas in range(self.num_classes):
            class_features = features[targets == clas]
            num_class_features = class_features.shape[0]
            # number of empty elements in the queue
            num_empty = self.queue_size - len(self.queue[clas])
            if(num_empty == 0):
                # empty the queue
                self.queue[clas] = []
            if num_class_features > num_empty:
                self.queue[clas].extend(class_features[:num_empty])
            elif num_class_features > 0 and (num_class_features <= num_empty):
                self.queue[clas].extend(class_features)

        # update the prototypes
        for clas in range(self.num_classes):
            if(len(self.queue[clas]) == self.queue_size):
                new = torch.mean(torch.stack(self.queue[clas]), dim=0).to(self.device)
                self.prototypes[clas] = self.momentum * self.prototypes[clas] + (1 - self.momentum) * new
                # empty the queue
                self.queue[clas] = []
    
    def get_prototypes(self):
        return self.prototypes
        
class BCLModel(nn.Module):
    def __init__(self, num_classes=1000, name='resnet50', head='mlp', use_norm=True, feat_dim=1024, recalibrate = False, beta = 0.99, initial_wc = 0.01, pretrained=False, cls_num_list = [], ema_prototypes = False, static = False):
        super(BCLModel, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(num_classes = num_classes, pretrained=pretrained)
        if head == 'mlp':
            self.head = nn.Sequential(nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
                                      nn.Linear(dim_in, feat_dim))
        else:
            raise NotImplementedError(
                'head not supported'
            )
        if use_norm:
            self.fc = NormedLinear(dim_in, num_classes)
        else:
            self.fc = nn.Linear(dim_in, num_classes)
        self.head_fc = nn.Sequential(nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
                                   nn.Linear(dim_in, feat_dim))
        self.recalibrate = recalibrate
        self.ema_prototypes = ema_prototypes
        if(self.ema_prototypes):
            self.prototype_store = PrototypeStore(num_classes, 2048, 'cuda')
        if(self.recalibrate):
            self.recalibrator = PrototypeRecalibrator(beta=beta, initial_wc=initial_wc, num_classes=num_classes, cls_num_list = cls_num_list, static = static)

    def forward(self, x, targets=None, phase='train'):
        feat = self.encoder(x)
        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat)
        if(self.ema_prototypes and phase =="train"):
            self.prototype_store.update_prototypes(feat, targets)
            centers_logits = F.normalize(self.head_fc(self.prototype_store.get_prototypes().type(torch.float32)), dim=1)
        else:
            centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1) # prototypes

        # TODO: recalibrate logits
        if(self.recalibrate and phase == 'train'):
            self.recalibrator.update(centers_logits, feat_mlp, targets) #update recalibrator
            centers_logits_calibrated = self.recalibrator.recalibrate(centers_logits) #recalibrate
            return feat_mlp, logits, centers_logits_calibrated
        return feat_mlp, logits, centers_logits
