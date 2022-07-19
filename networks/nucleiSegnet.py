import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import Constants

nonlinearity = partial(F.relu, inplace=True)

class Separable_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Separable_Block, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class Robust_Residual_Block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Robust_Residual_Block, self).__init__()

        self.conv1=nn.Conv2d(in_ch,in_ch,3,1,padding=1)
        self.norm1= nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.separable=Separable_Block(in_ch,in_ch)
        self.conv2=nn.Conv2d(in_ch,out_ch,3,1,padding=1)
        self.norm2= nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x1=self.conv1(x)
        x2=self.norm1(x1)
        x3=self.relu(x2)
        x4=self.separable(x3)
        x5=self.conv1(x4)
        x6=self.norm1(x5)
        x7=self.relu(x6)

        out=x+x7
        out=self.conv2(out)
        out=self.norm2(out)
        return F.relu(out)

class Bottle_Neck_Block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Bottle_Neck_Block, self).__init__()

        self.conv1=nn.Conv2d(in_ch,out_ch,3,1,padding=1)
        # self.conv1=nn.Conv2d(in_ch,out_ch,3,1,padding=1)
        self.norm1= nn.BatchNorm2d(out_ch)
        self.norm2= nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_ch,in_ch,3,1,padding=1)
    
    def forward(self, x):
        x1=self.conv1(x)
        x1=self.norm1(x1)
        x2=self.relu(x1)
        x2=self.conv2(x2)
        x2=self.norm2(x2)
        x3=self.relu(x2)
        x3=self.conv1(x3)
        x3=self.norm1(x3)
        out=self.relu(x3)
        return out

class Attention_Gate_Block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Attention_Gate_Block, self).__init__()  
        self.conv1=nn.Conv2d(out_ch,out_ch,3,1,padding=1) 
        self.conv2=nn.Conv2d(in_ch,out_ch,1,1)
        # self.conv3=nn.Conv2d(out_ch,out_ch,3,1,padding=1) 
        self.upsampling=nn.UpsamplingNearest2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.sig=nn.Sigmoid()
        self.conv3=nn.Conv2d(out_ch,out_ch//2,1,1)
        # self.norm1= nn.BatchNorm2d(out_ch)
        self.norm= nn.BatchNorm2d(out_ch//2)
    
    def forward(self, bottleneck, jump):
        x1=self.conv2(bottleneck)
        x1=self.conv1(x1)
        x1=self.upsampling(x1)

        x2=self.conv1(jump)
        x3=self.relu(x1+x2)
        x3=self.conv1(x3)
        x4=self.sig(x3)
        # x5=self.Resampler(jump.shape[2],jump.shape[3],x4)
        # x5=WeightedRandomSampler(x4, num_samples=1,replacement=True)


        x5=x4*jump
        x5=self.conv3(x5)
        x5=self.norm(x5)
        out=self.relu(x5)
        return out

    def Resampler(self,H,W,inp):
        new_h = torch.linspace(-1, 1, H).view(-1, 1).repeat(1, W)
        new_w = torch.linspace(-1, 1, W).repeat(H, 1)
        grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
        grid = grid.unsqueeze(0)
        outp = F.grid_sample(inp, grid=grid, mode='bilinear').cuda()
        return outp

class Attention_Decoder_Block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Attention_Decoder_Block, self).__init__()  
        self.relu = nn.ReLU(inplace=True)
        self.transpose=nn.ConvTranspose2d(in_ch, out_ch//2,2, stride=2)
        self.gate=Attention_Gate_Block(in_ch,out_ch)
    
    def forward(self, bottleneck, jump):
        x1=self.gate(bottleneck, jump)
        x2=self.transpose(bottleneck)
        x3=self.relu(x2)
        # out=x1+x3
        out=torch.cat((x3,x1), dim=1)
        return out

class Conv_Block(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(Conv_Block, self).__init__() 
        self.conv1=nn.Conv2d(in_ch,out_ch,3,1,padding=1) 
        self.conv2=nn.Conv2d(out_ch,out_ch,3,1,padding=1)
        self.norm= nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x1=self.conv1(x)
        x1=self.norm(x1)
        x2=self.relu(x1)
        x2=self.conv2(x2)
        x2=self.norm(x2)
        out=self.relu(x2)
        return out

class NucleiSegNet(nn.Module):
    def __init__(self, num_classes=Constants.BINARY_CLASS, num_channels=3):
        super(NucleiSegNet, self).__init__()   
        filters = [32, 64, 128, 256, 512]
        self.res1= Robust_Residual_Block(num_channels,filters[0])
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2= Robust_Residual_Block(filters[0],filters[1])
        self.res3= Robust_Residual_Block(filters[1],filters[2])
        self.res4= Robust_Residual_Block(filters[2],filters[3])
        self.neck=Bottle_Neck_Block(filters[3],filters[4])
        self.att1=Attention_Decoder_Block(filters[4],filters[3])
        self.conv1=Conv_Block(filters[3],filters[3])
        self.att2=Attention_Decoder_Block(filters[3],filters[2])
        self.conv2=Conv_Block(filters[2],filters[2])
        self.att3=Attention_Decoder_Block(filters[2],filters[1])
        self.conv3=Conv_Block(filters[1],filters[1])
        self.att4=Attention_Decoder_Block(filters[1],filters[0])
        self.conv4=Conv_Block(filters[0],filters[0])
        self.conv=nn.Conv2d(filters[0],num_classes,1,1)
    
    def forward(self, x):
        
        #down-sampling
        x1=self.res1(x)
        x2=self.pooling(x1)
        x3=self.res2(x2)
        x4=self.pooling(x3)
        x5=self.res3(x4)
        x6=self.pooling(x5)
        x7=self.res4(x6)
        x8=self.pooling(x7)

        #Bottle_Neck
        x9=self.neck(x8)

        #up-sampling
        y1=self.att1(x9,x7)
        y1=self.conv1(y1)
        y2=self.att2(y1,x5)
        y2=self.conv2(y2)
        y3=self.att3(y2,x3)
        y3=self.conv3(y3)
        y4=self.att4(y3,x1)
        y4=self.conv4(y4)

        out=self.conv(y4)
        return F.sigmoid(out)