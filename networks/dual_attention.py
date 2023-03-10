import torch
import torch.nn as nn

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialAttentionLayer(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttentionLayer, self).__init__()
        self.channelpool = ChannelPool()
        self.convlayer = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                                   padding=(kernel_size - 1) // 2)
        # self.relu = nn.ReLU(True)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # import pdb;pdb.set_trace()
        d = self.channelpool(x)
        d_out = self.convlayer(d)
        scale = torch.sigmoid(d_out)  # broadcasting
        return x * scale

class ChannelAttentionLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DualAttentionLayer(nn.Module):
    def __init__(self, num_feature=32, kernel_size=3, reduction=8):
        super(DualAttentionLayer, self).__init__()
        conv_list = [
            nn.Conv2d(num_feature, num_feature, kernel_size, padding=(kernel_size // 2),
                      bias=True, stride=1),
            nn.ReLU(True),
            nn.Conv2d(num_feature, num_feature, kernel_size, padding=(kernel_size // 2),
                      bias=True, stride=1)
        ]
        self.conv_pre= nn.Sequential(*conv_list)
        self.channel_att = ChannelAttentionLayer(num_feature, reduction)
        self.spatial_att = SpatialAttentionLayer()
        self.conv1x1 = nn.Conv2d(num_feature*2, num_feature, kernel_size=1)
    def forward(self, x):
        res = self.conv_pre(x)
        spatial = self.spatial_att(res)
        channel = self.channel_att(res)
        res = torch.cat([spatial, channel], dim=1)
        res = self.conv1x1(res)
        return res + x

class AttentionWeight(nn.Module):
    def __init__(self, channel=3, num_feature=96, kernel_size=3,num_img_in=3):
        super(AttentionWeight, self).__init__()
        self.num_img_in=num_img_in
        body_list = [DualAttentionLayer(num_feature=num_feature, kernel_size=3, reduction=8) for i in range(3)]
        self.body = nn.Sequential(*body_list)
        self.conv_down = nn.Conv2d(in_channels=channel*num_img_in, out_channels=num_feature, kernel_size=kernel_size, bias=True,
                                   stride=1, padding=1)
        self.conv_up = nn.Conv2d(in_channels=num_feature, out_channels=channel, kernel_size=kernel_size, bias=True,
                                   stride=1, padding=1)

    def forward(self, x, y, xk):
        x = self.conv_down(torch.cat([x, y, xk], 1))
        x = self.body(x)
        x = torch.tanh(self.conv_up(x))
        return x

class AttentionWeight2(nn.Module):
    def __init__(self, channel=3, num_feature=96, kernel_size=3,num_img_in=2):
        super(AttentionWeight2, self).__init__()
        self.num_img_in=num_img_in
        body_list = [DualAttentionLayer(num_feature=num_feature, kernel_size=3, reduction=8) for i in range(1)]
        self.body = nn.Sequential(*body_list)
        self.conv_down = nn.Conv2d(in_channels=channel*num_img_in, out_channels=num_feature, kernel_size=kernel_size, bias=True,
                                   stride=1, padding=1)
        self.conv_up = nn.Conv2d(in_channels=num_feature, out_channels=channel, kernel_size=kernel_size, bias=True,
                                   stride=1, padding=1)

    def forward(self, x, xk):
        x = self.conv_down(torch.cat([x, xk], 1))
        x = self.body(x)
        x = torch.tanh(self.conv_up(x))
        return x


class ConvLayer(nn.Module):
    def __init__(self,channel=3, num_feature=96, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=channel, out_channels=num_feature, kernel_size=kernel_size, stride=1, padding=1,
                                 bias=True)
        self.conv_2 = nn.Conv2d(in_channels=num_feature, out_channels=channel, kernel_size=kernel_size, stride=1, padding=1,
                                 bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        return x

class  TripleInput(nn.Module):
    def __init__(self, channel=3, num_feature=96, kernel_size=3):
        super(TripleInput, self).__init__()
        self.convlayer_x = ConvLayer(channel=channel,num_feature=num_feature,kernel_size=kernel_size)
        self.convlayer_y = ConvLayer(channel=channel, num_feature=num_feature, kernel_size=kernel_size)
        self.convlayer_xk = ConvLayer(channel=channel, num_feature=num_feature, kernel_size=kernel_size)
        self.down = nn.Conv2d(in_channels=channel * 3, out_channels=num_feature, kernel_size=kernel_size,
                                   bias=True,stride=1, padding=1)
    def forward(self,x,y,xk):
        x = self.convlayer_x(x)
        y = self.convlayer_y(y)
        xk = self.convlayer_xk(xk)
        x = self.down(torch.cat([x, y, xk], 1))
        return x

class TripleAttention(nn.Module):
    def __init__(self, channel=3, num_feature=96, kernel_size=3):
        super(TripleAttention, self).__init__()
        self.triple = TripleInput(num_feature=num_feature)
        body_list = [DualAttentionLayer(num_feature=num_feature, kernel_size=3, reduction=8) for i in range(1)]
        self.body = nn.Sequential(*body_list)
        # self.conv_down = nn.Conv2d(in_channels=channel * 3, out_channels=num_feature, kernel_size=kernel_size,bias=True, stride=1, padding=1)
        self.conv_up = nn.Conv2d(in_channels=num_feature, out_channels=channel, kernel_size=kernel_size, bias=True, stride=1, padding=1)

    def forward(self,x,y,xk):
        x = self.triple(x,y,xk)
        x = self.body(x)
        x = self.conv_up(x)
        return torch.tanh(x)
