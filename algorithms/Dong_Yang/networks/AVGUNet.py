import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
Softmax 0+1
'''

class Conv3d_Block(nn.Module):
    def __init__(self, num_in, num_out, kernel_size=1, stride=1, g=1, padding=None):
        super(Conv3d_Block, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
        self.Conv = nn.Sequential(
            nn.Conv3d(num_in, num_out, kernel_size=kernel_size, padding=padding, stride=stride, groups=g, bias=False),
            nn.BatchNorm3d(num_out),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):  # BN + Relu + Conv
        return self.Conv(x)


class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DoubleConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv3d(in_channel,out_channel,3,1,1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(out_channel,out_channel,3,1,1, bias=False),
            nn.BatchNorm3d(out_channel),
        )
        self.shortcut = nn.Conv3d(in_channel, out_channel, 1, 1, 0, bias=False)

    def forward(self, x):
        res=self.conv(x)
        residual=self.shortcut(x)
        res+=residual
        return nn.LeakyReLU(inplace=True)(res)


class AVGUNet(nn.Module):
    def __init__(self, in_ch, out_ch, sliding_window=False):
        super(AVGUNet, self).__init__()
        self.sw=sliding_window
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool3d((2))

        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Sequential(DoubleConv(256,256),DoubleConv(256,256))

        self.avg = skip_feature()

        self.up4 = UNetUp(256,256,256)
        self.up3 = UNetUp(256, 128, 128)
        self.up2 = UNetUp(128, 64, 64)
        self.up1 = UNetUp(64, 32, 32)

        self.outconv1= nn.Conv3d(32, out_ch, 1, 1, 0, bias=False)
        self.outconv2 = nn.Conv3d(64, out_ch, 1, 1, 0, bias=False)
        self.outconv3 = nn.Conv3d(128, out_ch, 1, 1, 0, bias=False)
        self.outconv4 = nn.Conv3d(256, out_ch, 1, 1, 0, bias=False)
        self.outconv5 = nn.Conv3d(256, out_ch, 1, 1, 0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data, 0.01)
                # nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, 0.01)
                # nn.init.constant_(m.bias.data, 0.0)
        self.count = 0

    def set_sw(self, symple):
        self.sw = symple

    def forward(self, x):
        # self.count += 1
        # print(self.count)

        size=x.size()[2:]
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        # print('p1', p1.shape) p1 torch.Size([1, 32, 24, 56, 56])
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # print('p2', p2.shape) p2 torch.Size([1, 64, 12, 28, 28])
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        # print('p3',p3.shape) p3 torch.Size([1, 128, 6, 14, 14])
        c4 = self.conv4(p3)
        # print('c4', c4.shape) c4 torch.Size([1, 256, 6, 14, 14])
        p4 = self.pool4(c4)
        # print('p4', p4.shape) p4 torch.Size([1, 256, 3, 7, 7])
        c5 = self.conv5(p4)
        # print(c5.shape, c4.shape, c3.shape, c2.shape, c1.shape)
        c4 = self.avg(c4)
        c3 = self.avg(c3)
        c2 = self.avg(c2)
        c1 = self.avg(c1)

        up4=self.up4(c5, c4)
        up3=self.up3(up4, c3)
        up2=self.up2(up3, c2)
        up1=self.up1(up2, c1)
        s1 = self.outconv1(up1)
        if self.sw:
            return s1
        else:
            s5=self.outconv5(c5)
            s4=self.outconv4(up4)
            s3=self.outconv3(up3)
            s2=self.outconv2(up2)
            return [F.interpolate(z, size, mode='trilinear', align_corners=False) for z in [s1, s2, s3, s4, s5]]


class UNetUp(nn.Module):
    def __init__(self, dec_ch, enc_ch, out_ch):
        super(UNetUp, self).__init__()
        self.conv = DoubleConv(enc_ch*2, out_ch)
        self.up = nn.ConvTranspose3d(dec_ch, out_ch, 2, stride=2, bias=False)

    def forward(self, x, y):
        x=self.up(x)
        return self.conv(torch.cat([x,y], 1))


class skip_feature(nn.Module):
    def __init__(self):
        super(skip_feature, self).__init__()
        self.avg = nn.AvgPool3d(3,stride=1)
        # self.conv = Conv3d_Block(enc_out, enc_out, 3, 1, 1)
    def forward(self,x):
        # print(x.shape)
        # x_avg = F.interpolate(self.avg(x), x.size()[2:], mode='trilinear', align_corners=False)
        x_avg = torch.nn.functional.pad(self.avg(x), (1,1,1,1,1,1,0,0,0,0), 'constant', value=0)
        # print('skip_feature', x.shape, self.avg(x).shape, x_avg.shape)
        x = x-x_avg
        return x

class test_transfer(nn.Module):
    def __init__(self, in_ch, out_ch, sliding_window=False):
        super(test_transfer, self).__init__()
        self.sw=sliding_window
        self.conv1 = DoubleConv(in_ch, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool3d((2))

        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool3d(2)

        self.conv5 = nn.Sequential(DoubleConv(256,256),DoubleConv(256,256))

        self.avg = skip_feature()

        self.up4 = UNetUp(256,256,256)
        self.up3 = UNetUp(256, 128, 128)
        self.up2 = UNetUp(128, 64, 64)
        self.up1 = UNetUp(64, 32, 32)

        self.outconv1= nn.Conv3d(32, out_ch, 1, 1, 0, bias=False)
        self.outconv2 = nn.Conv3d(64, out_ch, 1, 1, 0, bias=False)
        self.outconv3 = nn.Conv3d(128, out_ch, 1, 1, 0, bias=False)
        self.outconv4 = nn.Conv3d(256, out_ch, 1, 1, 0, bias=False)
        self.outconv5 = nn.Conv3d(256, out_ch, 1, 1, 0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data, 0.01)
                # nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data, 0.01)
                # nn.init.constant_(m.bias.data, 0.0)
        self.count = 0

    def forward(self, x):
        self.count += 1
        print(self.count)

        size=x.size()[2:]
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        # print('p1', p1.shape) p1 torch.Size([1, 32, 24, 56, 56])
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        # print('p2', p2.shape) p2 torch.Size([1, 64, 12, 28, 28])
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        # print('p3',p3.shape) p3 torch.Size([1, 128, 6, 14, 14])
        c4 = self.conv4(p3)
        # print('c4', c4.shape) c4 torch.Size([1, 256, 6, 14, 14])
        p4 = self.pool4(c4)
        # print('p4', p4.shape) p4 torch.Size([1, 256, 3, 7, 7])
        c5 = self.conv5(p4)
        # print(c5.shape, c4.shape, c3.shape, c2.shape, c1.shape)
        c4 = self.avg(c4)
        c3 = self.avg(c3)
        c2 = self.avg(c2)
        c1 = self.avg(c1)

        up4=self.up4(c5, c4)
        up3=self.up3(up4, c3)
        up2=self.up2(up3, c2)
        up1=self.up1(up2, c1)
        s1 = self.outconv1(up1)
        if self.sw:
            return s1
        else:
            s5=self.outconv5(c5)
            s4=self.outconv4(up4)
            s3=self.outconv3(up3)
            s2=self.outconv2(up2)
            return [F.interpolate(z, size, mode='trilinear', align_corners=False) for z in [s1, s2, s3, s4, s5]]


class demo_net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(demo_net, self).__init__()
        self.l1 = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)
        self.l2 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.l3 = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.l4 = nn.Conv2d(2, out_ch, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, 0.01)

    def forward(self ,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        return x



if __name__ == '__main__':
    # from thop import profile
    # x=torch.rand(2,1,96,96,96)
    # model = AVGUNet(1, 4, True)
    # # y = model(x)
    # # torch.Size([1, 8, 48, 112, 112]) torch.Size([1, 8, 48, 112, 112]) torch.Size([1, 8, 48, 112, 112]) torch.Size([1, 8, 48, 112, 112]) torch.Size([1, 8, 48, 112, 112])
    # # print([x.shape for x in y])
    # # total = sum([param.nelement() for param in model.parameters()])
    # #
    # # print("Number of parameter: %.2fM" % (total / 1e6))
    # flops, params = profile(model, (x,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # # flops: 454203.18 M, params: 18.86 M

    # test = test_transfer(1, 14, False)
    #
    # # 冻结层
    # for k ,v in test.named_parameters():
    #     if any(x in k.split('.') for x in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
    #         v.requires_grad = False
    #
    # print(test)
    # para = test.named_parameters()
    # for k, v in para:
    #     # print(f"{k}: \n{v}\n")
    #     print(f"{k}: require_grad is {v.requires_grad}\n")

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    test = test_transfer(1, 2, True)
    # 冻结层
    for k ,v in test.named_parameters():
        if any(x in k.split('.') for x in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']):
            v.requires_grad = False
    device = torch.device('cuda')
    test.to(device)
    print(test)
    para = test.named_parameters()
    for k, v in para:
        # print(f"{k}: \n{v}\n")
        print(f"{k}: require_grad is {v.requires_grad}\n")

    from torch.optim import SGD
    from torch.autograd import Variable
    from monai.losses import DiceLoss
    optimizer = SGD(filter(lambda p: p.requires_grad, test.parameters()), lr=1, momentum=0.9, weight_decay=0.01)
    # optimizer = SGD(test.parameters(), lr=1, momentum=0.9, weight_decay=0.01)

    ce = DiceLoss(softmax=True)

    label = np.ones((1, 2, 32, 32, 32)).astype(np.uint8)
    label = torch.from_numpy(label)
    label = Variable(label).cuda()

    import copy
    params_2 = copy.deepcopy(list(test.named_parameters()))
    # for a in params_2:
    #     print(a)

    for i in range(100):
        print(f'now is {i} cycle!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        data = np.random.randn(1, 1, 32, 32, 32)
        data = torch.from_numpy(data)

        data = Variable(data).float().cuda()

        output = test(data)

        loss = ce(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        params = list(test.named_parameters())
        for i in range(len(params)):
            # print(a)
            if params_2[i] != params and i < len(params)/2:
                print('error!!!')


