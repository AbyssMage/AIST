import torch
import torch.nn as nn
from collections import OrderedDict


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    # print(in_channels, out_channels)
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        # self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        # if self.has_bn:
        #    x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x


class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual  # short cut

        return out


class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x


# Matrix-based fusion
class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w),
                                    requires_grad=True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        x = x * self.weights  # element-wise multiplication

        return x


class stresnet(nn.Module):
    def __init__(self, c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32),
                 t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
        """
            C - Temporal Closeness
            P - Period
            T - Trend
            conf = (len_seq, number of flow - inflow & outflow, map_height, map_width)
            external_dim
        """

        super(stresnet, self).__init__()

        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf

        # self.nb_flow, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]
        self.nb_flow, self.map_height, self.map_width = t_conf[2], t_conf[3], t_conf[4]
        print(self.nb_flow, self.map_height, self.map_width)

        self.relu = torch.relu
        self.tanh = torch.tanh
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.qr_nums = len(self.quantiles)

        """self.c_way = self.make_one_way(in_channels=self.c_conf[0] * self.nb_flow)
        self.p_way = self.make_one_way(in_channels=self.p_conf[0] * self.nb_flow)
        self.t_way = self.make_one_way(in_channels=self.t_conf[0] * self.nb_flow)"""

        self.c_way = self.make_one_way(in_channels=self.c_conf[1] * self.nb_flow)
        self.p_way = self.make_one_way(in_channels=self.p_conf[1] * self.nb_flow)
        self.t_way = self.make_one_way(in_channels=self.t_conf[1] * self.nb_flow)

        self.external_ops = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.external_dim, 10, bias=True)),
            ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, self.nb_flow * self.map_height * self.map_width, bias=True)),
            ('relu2', nn.ReLU()),
        ]))

    def make_one_way(self, in_channels):
        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels=in_channels, out_channels=64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter=64, repetations=self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels=64, out_channels=2)),
            ('FusionLayer', TrainableEltwiseLayer(n=self.nb_flow, h=self.map_height, w=self.map_width))
        ]))

    def forward(self, input_c, input_p, input_t, input_ext):
        # Three-way Convolution
        main_output = 0
        bs = 42

        # input_c = input_c.view(-1, self.c_conf[0] * 2, self.map_height, self.map_width)
        input_c = input_c.view(bs, -1, self.c_conf[1] * 2, self.map_height, self.map_width)
        out_c = self.c_way(input_c)
        main_output += out_c
        print(main_output.shape)

        # input_p = input_p.view(-1, self.p_conf[0] * 2, self.map_height, self.map_width)
        input_p = input_p.view(bs, -1, self.p_conf[1] * 2, self.map_height, self.map_width)
        out_p = self.p_way(input_p)
        main_output += out_p
        print(main_output.shape)

        # input_t = input_t.view(-1, self.t_conf[0] * 2, self.map_height, self.map_width)
        input_t = input_t.view(bs, -1, self.t_conf[1] * 2, self.map_height, self.map_width)
        out_t = self.t_way(input_t)
        main_output += out_t
        print(main_output.shape)

        # parameter-matrix-based fusion
        # main_output = out_c + out_p + out_t

        external_output = self.external_ops(input_ext)
        external_output = self.relu(external_output)
        external_output = external_output.view(-1, self.nb_flow, self.map_height, self.map_width)
        # main_output = torch.add(main_output, external_output)
        main_output += external_output
        print(main_output.shape)

        main_output = self.tanh(main_output)
        return main_output


if __name__ == '__main__':

    """x1 = torch.randn([3, 2, 16, 8])
    x2 = torch.randn([4, 2, 16, 8])
    x3 = torch.randn([4, 2, 16, 8])
    x4 = torch.randn([8])"""

    bs = 42
    x1 = torch.randn([bs, 3, 2, 16, 8])
    x2 = torch.randn([bs, 4, 2, 16, 8])
    x3 = torch.randn([bs, 4, 2, 16, 8])
    x4 = torch.randn([bs, 8])

    model = stresnet([bs, 3, 2, 16, 8], [bs, 4, 2, 16, 8], [bs, 4, 2, 16, 8], external_dim=8, nb_residual_unit=4)
    output = model(x1, x2, x3, x4)

    print(model)
    print(output.shape)
