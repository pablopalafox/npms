import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nnutils import make_conv, ResBlock


class ShapeEncoder(nn.Module):

    def __init__(self, code_dim=256, res=128):
        super(ShapeEncoder, self).__init__()

        fn_0 = 1
        fn_1 = 8
        fn_2 = 16
        fn_3 = 32
        fn_4 = 64
        fn_5 = 128
        fn_6 = 256

        self.conv_in  = make_conv(fn_0, fn_1, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        
        self.conv_0   = make_conv(fn_1, fn_2, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        self.conv_0_1 = ResBlock(fn_2, normalization=nn.BatchNorm3d)
        
        self.conv_1   = make_conv(fn_2, fn_3, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        self.conv_1_1 = ResBlock(fn_3, normalization=nn.BatchNorm3d)
        
        self.conv_2   = make_conv(fn_3, fn_4, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        self.conv_2_1 = ResBlock(fn_4, normalization=nn.BatchNorm3d)
        
        if res == 128:
            self.conv_3   = make_conv(fn_4, fn_5, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
            self.conv_3_1 = nn.Conv3d(fn_5, fn_5, 3, padding=1, padding_mode='replicate')

            self.fc_0   = nn.Conv1d(fn_5 * 4**3, code_dim * 2, 1)

        elif res == 256:
            self.conv_3   = make_conv(fn_4, fn_5, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
            self.conv_3_1 = ResBlock(fn_5, normalization=nn.BatchNorm3d)

            self.conv_4   = make_conv(fn_5, fn_6, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
            self.conv_4_1 = nn.Conv3d(fn_6, fn_6, 3, padding=1, padding_mode='replicate')

            self.fc_0   = nn.Conv1d(fn_6 * 4**3, code_dim * 2, 1)
        
        else:
            exit()
        
        self.fc_out = nn.Conv1d(code_dim * 2, code_dim, 1)
        self.actvn  = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        bs = x.shape[0]
        x = x.unsqueeze(1) # [bs, 1, res, res, res]

        net = self.conv_in(x)

        net = self.conv_0(net)
        net = self.conv_0_1(net)

        net = self.conv_1(net)
        net = self.conv_1_1(net)

        net = self.conv_2(net)
        net = self.conv_2_1(net)

        net = self.conv_3(net)
        net = self.conv_3_1(net)

        if hasattr(self, 'conv_4'):
            net = self.conv_4(net)
            net = self.conv_4_1(net)
        
        y = net.view(bs, -1, 1) # bs 512 1

        y = self.actvn(self.fc_0(y))
        y = self.fc_out(y)
        y = y.permute(0, -1, 1)
        return y


class PoseEncoder(nn.Module):

    def __init__(self, code_dim=256, res=128):
        super(PoseEncoder, self).__init__()

        fn_0 = 1
        fn_1 = 16
        fn_2 = 32
        fn_3 = 64
        fn_4 = 128
        fn_5 = 256
        fn_6 = 256

        self.conv_in  = make_conv(fn_0, fn_1, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        
        self.conv_0   = make_conv(fn_1, fn_2, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        self.conv_0_1 = ResBlock(fn_2, normalization=nn.BatchNorm3d)
        
        self.conv_1   = make_conv(fn_2, fn_3, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        self.conv_1_1 = ResBlock(fn_3, normalization=nn.BatchNorm3d)
        
        self.conv_2   = make_conv(fn_3, fn_4, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
        self.conv_2_1 = ResBlock(fn_4, normalization=nn.BatchNorm3d)
        
        if res == 128:
            self.conv_3   = make_conv(fn_4, fn_5, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
            self.conv_3_1 = nn.Conv3d(fn_5, fn_5, 3, padding=1, padding_mode='replicate')

            self.fc_0   = nn.Conv1d(fn_5 * 4**3, code_dim * 2, 1)

        elif res == 256:
            self.conv_3   = make_conv(fn_4, fn_5, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
            self.conv_3_1 = ResBlock(fn_5, normalization=nn.BatchNorm3d)

            self.conv_4   = make_conv(fn_5, fn_6, n_blocks=1, kernel=3, stride=2, normalization=nn.BatchNorm3d)
            self.conv_4_1 = nn.Conv3d(fn_6, fn_6, 3, padding=1, padding_mode='replicate')

            self.fc_0   = nn.Conv1d(fn_6 * 4**3, code_dim * 2, 1)
        
        else:
            exit()
        
        self.fc_out = nn.Conv1d(code_dim * 2, code_dim, 1)
        self.actvn  = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None:
                    nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        bs = x.shape[0]
        x = x.unsqueeze(1) # [bs, 1, res, res, res]

        net = self.conv_in(x)

        net = self.conv_0(net)
        net = self.conv_0_1(net)

        net = self.conv_1(net)
        net = self.conv_1_1(net)

        net = self.conv_2(net)
        net = self.conv_2_1(net)

        net = self.conv_3(net)
        net = self.conv_3_1(net)

        if hasattr(self, 'conv_4'):
            net = self.conv_4(net)
            net = self.conv_4_1(net)
        
        y = net.view(bs, -1, 1) # bs 512 1

        y = self.actvn(self.fc_0(y))
        y = self.fc_out(y)
        y = y.permute(0, -1, 1)
        return y
