import torch
import torch.nn as nn
import torch.nn.functional as F


class demoireing_net(nn.Module):
    def __init__(self, feat_num=48, inter_feat_num=32, shuffle_num=4):
        super(demoireing_net, self).__init__()
        self.affine = True
        self.feat_num = feat_num
        self.inter_feat_num = inter_feat_num
        self.shuffle_num = shuffle_num

        # preprocessing
        self.conv_pre = nn.Sequential(
            nn.Conv2d(3 * self.shuffle_num * self.shuffle_num, feat_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )

        self.ddb_pre = DDB(in_channel=feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num)

        # down 1/4
        self.down_2 = nn.Sequential(
            nn.Conv2d(2 * feat_num, 3 * feat_num, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # down 1/2
        self.down_1 = nn.Sequential(
            nn.Conv2d(feat_num, 2 * feat_num, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # down 1/1
        # nothing to do

        # MGRB 1/4
        self.ddb_2 = DDB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num)
        self.mgrb_block_2_1 = MGRB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=self.affine, num_heads=8)
        self.mgrb_block_2_2 = MGRB(in_channel=3 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=self.affine, num_heads=8)

        # MGRB 1/2
        self.ddb_1 = DDB(in_channel=2 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num)
        self.mgrb_block_1_1 = MGRB(in_channel=2 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=self.affine, num_heads=6)
        self.mgrb_block_1_2 = MGRB(in_channel=2 * feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=self.affine, num_heads=6)

        # MGRB 1/1
        self.ddb_0 = DDB(in_channel=feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num)
        self.mgrb_block_0_1 = MGRB(in_channel=feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=self.affine, num_heads=4)
        self.mgrb_block_0_2 = MGRB(in_channel=feat_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num, affine=self.affine, num_heads=4)

        # 1/4
        # preconv
        self.preconv_2_moire = conv_relu(int((3 * feat_num) / 3 * 1), 2 * feat_num, 1, padding=0)
        self.preconv_2_clean = conv_relu(int((3 * feat_num) / 3 * 2), 2 * feat_num, 1, padding=0)

        # net heads
        self.net_head_2_moire = nn.Sequential(
            DB(in_channel=int((3 * feat_num) / 3 * 1), d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num),
        )
        self.net_head_2_clean = nn.Sequential(
            DB(in_channel=int((3 * feat_num) / 3 * 2), d_list=(1, 2, 3, 2, 1), inter_num=inter_feat_num),
        )

        # reconstruction
        self.recon_2_moire = conv(in_channel=int((3 * feat_num) / 3 * 1), out_channel=3 * self.shuffle_num * self.shuffle_num, kernel_size=3, padding=1)
        self.recon_2_clean = conv(in_channel=int((3 * feat_num) / 3 * 2), out_channel=3 * self.shuffle_num * self.shuffle_num, kernel_size=3, padding=1)

        # combine 1/4 and 1/2
        self.ftb_1 = FTB(2 * feat_num, 2 * feat_num)

        # preconv
        self.preconv_1_moire = conv_relu(int((2 * feat_num) / 3 * 1), feat_num, 1, padding=0)
        self.preconv_1_clean = conv_relu(int((2 * feat_num) / 3 * 2), feat_num, 1, padding=0)

        # net heads
        self.net_head_1_moire = nn.Sequential(
            DB(in_channel=int((2 * feat_num) / 3 * 1), d_list=(1, 2, 1), inter_num=inter_feat_num),
        )
        self.net_head_1_clean = nn.Sequential(
            DB(in_channel=int((2 * feat_num) / 3 * 2), d_list=(1, 2, 1), inter_num=inter_feat_num),
        )

        # reconstruction
        self.recon_1_moire = conv(in_channel=int((2 * feat_num) / 3 * 1), out_channel=3 * self.shuffle_num * self.shuffle_num, kernel_size=3, padding=1)
        self.recon_1_clean = conv(in_channel=int((2 * feat_num) / 3 * 2), out_channel=3 * self.shuffle_num * self.shuffle_num, kernel_size=3, padding=1)

        # combine 1/2 and 1
        self.ftb_0 = FTB(feat_num, feat_num)

        # net heads
        self.net_head_0_moire = nn.Sequential(
            DB(in_channel=int((1 * feat_num) / 3 * 1), d_list=(1, 1), inter_num=inter_feat_num),
        )
        self.net_head_0_clean = nn.Sequential(
            DB(in_channel=int((1 * feat_num) / 3 * 2), d_list=(1, 1), inter_num=inter_feat_num),
        )

        # reconstruction
        self.recon_0_moire = conv(in_channel=int((1 * feat_num) / 3 * 1), out_channel=3 * self.shuffle_num * self.shuffle_num, kernel_size=3, padding=1)
        self.recon_0_clean = conv(in_channel=int((1 * feat_num) / 3 * 2), out_channel=3 * self.shuffle_num * self.shuffle_num, kernel_size=3, padding=1)

        self.pre_head_0 = conv(in_channel=int((1 * feat_num) / 3 * 2), out_channel=int((1 * feat_num) / 3 * 1), kernel_size=1, padding=0)
        self.pre_head_1 = conv(in_channel=int((2 * feat_num) / 3 * 2), out_channel=int((2 * feat_num) / 3 * 1), kernel_size=1, padding=0)
        self.pre_head_2 = conv(in_channel=int((3 * feat_num) / 3 * 2), out_channel=int((3 * feat_num) / 3 * 1), kernel_size=1, padding=0)        

    def forward(self, x):
        # preprocessing
        x = F.pixel_unshuffle(x, self.shuffle_num)
        x = self.conv_pre(x)
        x = self.ddb_pre(x)
       
        # down 1/2
        x_down_1 = self.down_1(x)
        # down 1/4
        x_down_2 = self.down_2(x_down_1)

        # share compression 1/4
        x_down_2 = self.ddb_2(x_down_2)
        x_down_2_out = self.mgrb_block_2_1(x_down_2)
        x_down_2_out = self.mgrb_block_2_2(x_down_2_out)

        # split
        x_down_2_out_clean, x_down_2_out_moire = torch.split(x_down_2_out, split_size_or_sections=[int((3 * self.feat_num) / 3 * 2), int((3 * self.feat_num) / 3 * 1)], dim=1)

        # fusion: combine 1/4 and 1/2
        x_down_2_out_clean_up = self.preconv_2_clean(x_down_2_out_clean)
        x_down_2_out_clean_up = F.interpolate(x_down_2_out_clean_up, scale_factor=2, mode='bilinear')
        x_down_2_out_moire_up = self.preconv_2_moire(x_down_2_out_moire)
        x_down_2_out_moire_up = F.interpolate(x_down_2_out_moire_up, scale_factor=2, mode='bilinear')
    
        # ftb
        x_down_1_g, _, _ = self.ftb_1((x_down_1, x_down_2_out_moire_up), self.affine, return_shift=True)
        x_down_1_g = x_down_1_g + x_down_2_out_clean_up
        
        # net heads
        x_down_2_out_moire_head_ = self.net_head_2_moire(x_down_2_out_moire)
        x_down_2_out_clean_head_ = self.net_head_2_clean(x_down_2_out_clean)

        # reconstruction
        x_down_2_out_moire_head = self.recon_2_moire(x_down_2_out_moire_head_)
        x_down_2_out_moire_head = F.pixel_shuffle(x_down_2_out_moire_head, self.shuffle_num)
        x_down_2_out_clean_head = self.recon_2_clean(x_down_2_out_clean_head_)
        x_down_2_out_clean_head = F.pixel_shuffle(x_down_2_out_clean_head, self.shuffle_num)
        
        # share compression 1/2
        x_down_1_g_out = self.ddb_1(x_down_1_g)
        x_down_1_out = self.mgrb_block_1_1(x_down_1_g_out)
        x_down_1_out = self.mgrb_block_1_2(x_down_1_out)

        # split
        x_down_1_out_clean, x_down_1_out_moire = torch.split(x_down_1_out, split_size_or_sections=[int((2 * self.feat_num) / 3 * 2), int((2 * self.feat_num) / 3 * 1)], dim=1)

        # fusion: combine 1/2 and 1/1
        x_down_1_out_clean_up = self.preconv_1_clean(x_down_1_out_clean)
        x_down_1_out_clean_up = F.interpolate(x_down_1_out_clean_up, scale_factor=2, mode='bilinear')
        x_down_1_out_moire_up = self.preconv_1_moire(x_down_1_out_moire)
        x_down_1_out_moire_up = F.interpolate(x_down_1_out_moire_up, scale_factor=2, mode='bilinear')

        # ftb
        x_g, _, _ = self.ftb_0((x, x_down_1_out_moire_up), self.affine, return_shift=True)
        x_g = x_g + x_down_1_out_clean_up

        # net heads
        x_down_1_out_moire_head_ = self.net_head_1_moire(x_down_1_out_moire)
        x_down_1_out_clean_head_ = self.net_head_1_clean(x_down_1_out_clean)

        # reconstruction
        x_down_1_out_moire_head = self.recon_1_moire(x_down_1_out_moire_head_)
        x_down_1_out_moire_head = F.pixel_shuffle(x_down_1_out_moire_head, self.shuffle_num)
        x_down_1_out_clean_head = self.recon_1_clean(x_down_1_out_clean_head_)
        x_down_1_out_clean_head = F.pixel_shuffle(x_down_1_out_clean_head, self.shuffle_num)

        # share compression 1
        x_g_out = self.ddb_0(x_g)
        x_out = self.mgrb_block_0_1(x_g_out)
        x_out = self.mgrb_block_0_2(x_out)

        # split
        x_out_clean, x_out_moire = torch.split(x_out, split_size_or_sections=[int((1 * self.feat_num) / 3 * 2), int((1 * self.feat_num) / 3 * 1)], dim=1)

        # net heads
        x_out_moire_head_ = self.net_head_0_moire(x_out_moire)
        x_out_clean_head_ = self.net_head_0_clean(x_out_clean)

        # reconstruction
        x_out_moire_head = self.recon_0_moire(x_out_moire_head_)
        x_out_moire_head = F.pixel_shuffle(x_out_moire_head, self.shuffle_num)
        x_out_clean_head = self.recon_0_clean(x_out_clean_head_)
        x_out_clean_head = F.pixel_shuffle(x_out_clean_head, self.shuffle_num)

        # 3 clean and 3 moire images
        return x_out_clean_head, x_down_1_out_clean_head, x_down_2_out_clean_head, \
                x_out_moire_head, x_down_1_out_moire_head, x_down_2_out_moire_head
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class FTB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FTB, self).__init__()
        self.FTB_scale_conv0 = nn.Conv2d(in_channel, in_channel, 1)
        self.FTB_scale_conv1 = nn.Conv2d(in_channel, out_channel, 1)
        self.FTB_shift_conv0 = nn.Conv2d(in_channel, in_channel, 1)
        self.FTB_shift_conv1 = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x, affine=True, return_shift=False):
        if affine:
            # x[0]: fea; x[1]: cond
            scale = F.adaptive_avg_pool2d(self.FTB_scale_conv1(F.leaky_relu(self.FTB_scale_conv0(x[1]), 0.1, inplace=True)), (1, 1))
            shift = self.FTB_shift_conv1(F.leaky_relu(self.FTB_shift_conv0(x[1]), 0.1, inplace=True))

            if return_shift:
                return F.instance_norm(x[0] + shift) * scale, shift, F.instance_norm(shift)
            else:
                return F.instance_norm(x[0] + shift) * scale
        else:
            return x[0]


class MGRB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num, affine, num_heads):
        super(MGRB, self).__init__()

        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion_2 = CAB(3 * in_channel)
        self.ftb_2 = FTB(in_channel, in_channel)
        self.fusion_4 = CAB(3 * in_channel)
        self.ftb_4 = FTB(in_channel, in_channel)

        self.affine = affine

    def forward(self, x):
        x_0 = x

        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        y_4 = self.basic_block_4(x_4)
        y_4 = F.interpolate(y_4, scale_factor=2, mode='bilinear')

        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        y_2 = self.basic_block_2(self.fusion_4(self.ftb_4((x_2, y_4), self.affine), y_4, x_2))
        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')

        y_0 = self.basic_block(self.fusion_2(self.ftb_2((x_0, y_2), self.affine), y_2, x_0))

        y = x + y_0

        return y


class CAB(nn.Module):
    def __init__(self, in_chnls, ratio=8):
        super(CAB, self).__init__()
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = DB(in_channel=in_chnls // ratio, d_list=(1, 2, 3, 2, 1), inter_num=in_chnls // ratio)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x0, x1, x2):
        out_original = torch.cat([x0, x1, x2], dim=1)
        out = self.compress1(out_original)
        out = F.leaky_relu(out, 0.1, inplace=True)
        out = self.compress2(out)
        out = F.leaky_relu(out, 0.1, inplace=True)
        out = self.excitation(out)
        out = out + out_original
        out = self.squeeze(out)
        out = out.view(out.size(0), 3, -1, 1)
        out = F.softmax(out, dim=1)
        out = out.view(out.size(0), -1, 1, 1)
        w0, w1, w2 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x1 * w1 + x2 * w2

        return x
    

class DDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out
