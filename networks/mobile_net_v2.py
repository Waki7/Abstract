import logging

import torch
from torch import nn

from networks.net_factory import register_network
from networks.network_interface import NetworkInterface


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@register_network
class MobileNetV2(NetworkInterface):
    '''
    Dummy config

    image_encoder:
        lr: .003
        gradient_clip: 2.5

        weights_path: 'networks/weights/mobilenet_v2-b0353104.pth'
        last_block: -1
        max_block_repeats: 1
        pretrained: true
        out_features: 48

    '''

    def __init__(self, in_shapes, out_shapes, cfg={}):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__(in_shapes=in_shapes, out_shapes=out_shapes, cfg=cfg)
        block = InvertedResidual
        input_channel = 32
        default_last_channel = 1280

        self.in_channels = in_shapes[0][-3]
        out_features = self.out_features  # 1000
        self.max_block_repeats = cfg.get('max_block_repeats', 4)
        self.last_block = cfg.get('last_block', -1)
        width_mult = cfg.get('width_mult', 1.0)
        round_nearest = cfg.get('round_nearest', 8)

        # will initialize with this because that's what the pretrained weights use,
        # will be pruned to match the configured inverted_residual_setting
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
        logging.info('inverted residual setting is:\n{}'.format(inverted_residual_setting))

        last_block_num = len(inverted_residual_setting) + 1 + self.last_block
        self.excluded_weight_names = set()
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(default_last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(self.in_channels, input_channel, stride=2)]
        # building inverted residual blocks
        for block_num, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                if last_block_num == block_num or i > (self.max_block_repeats - 1):
                    for key in features[-1].state_dict().keys():
                        self.excluded_weight_names.add('features.{}.{}'.format(len(features) - 1, key))
        modified_last_channel = default_last_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, default_last_channel, kernel_size=1))

        if self.last_block < -1:
            modified_last_channel = inverted_residual_setting[self.last_block+1][1]  # is the index for out_channels
            for key in features[-1].state_dict().keys():
                self.excluded_weight_names.add('features.{}.{}'.format(len(features) - 1, key))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(modified_last_channel, out_features),
        )
        for key in self.classifier.state_dict().keys():
            self.excluded_weight_names.add('classifier.{}'.format(key))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.create_optimizer()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

    def load(self, weights_path):
        new_dict :dict= torch.load(weights_path)

        old_dict = self.state_dict()
        weights_to_keep = {}
        first_key = list(new_dict.keys())[0]
        first_weight: torch.Tensor = new_dict[first_key]

        assert len(first_weight.shape) == 4 and first_weight.shape[1] == 3, \
            'expecting first weight to be a convolution weight for 3 channels'
        if self.in_channels != first_weight.shape[1]:
            new_dict[first_key] = first_weight.mean(dim=1, keepdim=True)

        key_list = list(new_dict.keys())
        for key in key_list:
            if key not in self.excluded_weight_names:
                weights_to_keep[key] = new_dict[key]
            else:
                del new_dict[key]
                del old_dict[key]

        self.load_state_dict(weights_to_keep)
