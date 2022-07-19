'''attContextSeg in PyTorch.

See the paper "" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import math 
from functools import partial
from networks.utils import *
# import simclr
nonlinearity = partial(F.relu, inplace=True)

class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x1,x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x1.size()
        proj_query = self.query_conv(x1).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x2).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x1).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x1
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

    def forward(self, p, z):
        z = z.detach()
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return -(p * z).sum(dim=1).mean()
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #resnet18 = models.resnet18(pretrained=True)
        proj_hid, proj_out = 2048, 2048
        pred_hid, pred_out = 64, 2048
        self.backbone=AL_NET_encoder()
     
        checkpoint = torch.load('/home/zhaojing/AL-Net/weights/AL_NET_clusteredCell1en.tar')
        model_dict=checkpoint['model_state_dict']
        for k in list(model_dict.keys()):
            new_k=k.replace("module.","")
            model_dict[new_k]=model_dict.pop(k)
            # 载入参数
        self.backbone.load_state_dict(model_dict)
        self.decoder=AL_NET_decoder()

        checkpoint1 = torch.load('/home/zhaojing/AL-Net/weights/AL_NET_clusteredCell1de.tar')
        model_dict1=checkpoint1['model_state_dict']
        for k in list(model_dict1.keys()):
            new_k=k.replace("module.","")
            model_dict1[new_k]=model_dict1.pop(k)
            # 载入参数
        self.decoder.load_state_dict(model_dict1)
        
        # self.backbone = AL_NET_encoder()
        backbone_in_channels = 64

        self.projection = nn.Sequential(
            nn.Linear(backbone_in_channels, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_out),
            nn.BatchNorm1d(proj_out)
        )

        self.prediction = nn.Sequential(
            nn.Linear(proj_out, pred_hid),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(),
            nn.Linear(pred_hid, pred_out),
        )

        self.d = D()

    def forward(self, x1, x2):
        e1,ee1,e2,e3,e4,e5=self.backbone(x1)
        _,_,out11=self.decoder(e1,ee1,e2,e3,e4,e5)
        out1 =out11.squeeze()
        z1 = self.projection(out1)
        p1 = self.prediction(z1)

        w1,ww1,w2,w3,w4,w5=self.backbone(x2)
        _,_,out22=self.decoder(w1,ww1,w2,w3,w4,w5)
        out2 =out22.squeeze()
        # out2 = self.backbone(x2).squeeze()
        z2 = self.projection(out2)
        p2 = self.prediction(z2)

        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.
        
        return d1,d2
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c=channel // reduction
        if channel==3:
            self.c=3
        
        self.fc = nn.Sequential(
            nn.Linear(channel,self.c, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x* y.expand_as(x)

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_in // 4, 1)
        self.norm1 = nn.BatchNorm2d(ch_in // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(ch_in // 4, ch_in // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(ch_in // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(ch_in // 4, ch_out, 1)
        self.norm3 = nn.BatchNorm2d(ch_out)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class contextCode(nn.Module):
    def __init__(self, channel):
        super(contextCode, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        # self.Conv_1x1 = nn.Conv2d(channel,1,kernel_size=1,stride=1,padding=0 )   
        self.se=SELayer(channel=channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))

        # d1=self.Conv_1x1(dilate1_out)
        # d2=self.Conv_1x1(dilate2_out)
        # d3=self.Conv_1x1(dilate3_out)
        # d4=self.Conv_1x1(dilate4_out)
        # d12 = torch.cat([d1, d2], dim=1)
        # d123 = torch.cat([d12, d3], dim=1)
        # d1234 = torch.cat([d123, d4], dim=1)
        # se1234=self.se(d1234)
        # w1,w2,w3,w4=torch.split(se1234,1,1)
        # out = x + w1*dilate1_out + w2*dilate2_out + w3*dilate3_out + w4*dilate4_out
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        out=x+self.se(out)
        return out  
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
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

class dconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):  
        super(dconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            hswish(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            hswish(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            hswish()                     
        )
        self.cov11=nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,bias=True, dilation=2)
        self.conCode=contextCode(channel=ch_in)

    def forward(self,x):
        res=self.cov11(x)
        x=self.conCode(x)
        x = self.conv(x)
       
        return x+res
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))  
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)          
        )
     

    def forward(self,x):
        x = self.conv(x)
        return x
class cconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(cconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)            
        )
     

    def forward(self,x):
        x = self.conv(x)
        return x

class attentionLearn(nn.Module):
    def __init__(self, s):
        super(attentionLearn, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsam= nn.Upsample(scale_factor=2)
        n1 = 16 
        filters = [n1, n1 * 2, n1 * 4]        
        self.Conv1 = conv_block(ch_in=s,ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0],ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1],ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2],ch_out=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])
    
        self.Up2 = up_conv(ch_in=filters[1],ch_out=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
    
        self.Up1 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.Conv_1x1 = nn.Conv2d(filters[0],1,kernel_size=1,stride=1,padding=0 )       

    def forward(self, x):
      
        xup=self.upsam(x)
        
        x1 = self.Conv1(xup)
        x1=self.Maxpool(x1)
       
        x2 = self.Conv2(x1)
        x2=self.Maxpool(x2)
          
        x3 = self.Conv3(x2)
        x3=self.Maxpool(x3)
        d3 = self.Up3(x3)
     
        d3 = torch.cat((x2,d3),dim=1)
           
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)

        d2 = self.Up_conv2(d2)
        d2=self.Up1(d2)
        d1 = self.Conv_1x1(d2)
        rd=self.Maxpool(d1)
        psi=torch.sigmoid(rd) 
        return x+x*psi,psi
def unfreeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True
class AL_NET_k(nn.Module):#alccnet or no pretrain
    def __init__(self,img_ch=3, numclass=1):
        super(AL_NET_k, self).__init__()
         
        n1 = 64 #28
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4 
        self.conCode1=contextCode(channel=filters[0])
        self.conCode2=contextCode(channel=filters[0])
        self.conCode3=contextCode(channel=filters[1])
        self.conCode4=contextCode(channel=filters[2])
 
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.up= nn.Upsample(scale_factor=2)
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[2])
        self.att5=attentionLearn(s=filters[2])        
        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[2])
    
        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[1])
        self.att4=attentionLearn(s=filters[1])
       
        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[1])
    
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[0])
        self.att3=attentionLearn(s=filters[0])
        
        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[0])
    
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.att2=attentionLearn(s=filters[0])
       
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        
        self.Up1 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.sa512 = PAM_Module(512)
        self.sc512 = CAM_Module(512)
        self.model=Model() 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Conv3_1x1 = nn.Conv2d(filters[2],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv2_1x1 = nn.Conv2d(filters[1],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv1_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv0_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # encoding path
        x,xk=torch.split(x,3,1)
        x18 = self.firstconv(xk)
        x18 = self.firstbn(x18)
        x18 = self.firstrelu(x18)
        x18 = self.firstmaxpool(x18)
        e118 = self.encoder1(x18)
        e218= self.encoder2(e118)
        e318 = self.encoder3(e218)
        e418 = self.encoder4(e318)

        x = self.firstconv(x)
        x = self.firstbn(x)
        xx1 = self.firstrelu(x)
        x1 = self.firstmaxpool(xx1)
        x1=self.conCode1(x1)
        x2 = self.encoder1(x1)
        x2=self.conCode2(x2)
        x3 = self.encoder2(x2)
        x3=self.conCode3(x3)
        x4 = self.encoder3(x3)
        x4=self.conCode4(x4)
        x5 = self.encoder4(x4)        
     
        x55=  self.avgpool(x5)
        x418=  self.avgpool(e418)
        # d=self.model(x55,x418)
        
        x5=self.sa512(x5,x5)
        x5=self.sc512(x5)
        d5 = self.Up5(x5)
        xx4,at5=self.att5(x4)       
        d5 = torch.cat((xx4,d5),dim=1) 
        # d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)  
        
        d4 = self.Up4(d5)
        xx3,at4=self.att4(x3)
        d4 = torch.cat((xx3,d4),dim=1)
        # d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        xx2,at3=self.att3(x2)
        d3 = torch.cat((xx2,d3),dim=1)
        # d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1,at2=self.att2(xx1)
        d2 = torch.cat((x1,d2),dim=1)
        # d2 = torch.cat((xx1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        # att=self.Conv0_1x1(self.up(xx1))+self.Conv1_1x1(self.up(self.up(x2)))+self.Conv2_1x1(self.up(self.up(self.up(x3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(x4)))))
        att=self.Conv0_1x1(self.up(x1))+self.Conv1_1x1(self.up(self.up(xx2)))+self.Conv2_1x1(self.up(self.up(self.up(xx3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(xx4)))))
        d1 = self.Conv_1x1(d1)
        
        return torch.sigmoid(d1),torch.sigmoid(att)#,d
class AL_NET_encoder(nn.Module):#alccnet or no pretrain
    def __init__(self,img_ch=3, numclass=1):
        super(AL_NET_encoder, self).__init__()
         
        n1 = 64 #28
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4 
        self.conCode1=contextCode(channel=filters[0])
        self.conCode2=contextCode(channel=filters[0])
        self.conCode3=contextCode(channel=filters[1])
        self.conCode4=contextCode(channel=filters[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    

    def forward(self, x):
        # encoding path
        x = self.firstconv(x)
        x = self.firstbn(x)
        xx1 = self.firstrelu(x)
        x1 = self.firstmaxpool(xx1)
        x1=self.conCode1(x1)
        x2 = self.encoder1(x1)
        x2=self.conCode2(x2)
        x3 = self.encoder2(x2)
        x3=self.conCode3(x3)
        x4 = self.encoder3(x3)
        x4=self.conCode4(x4)
        x5 = self.encoder4(x4) 
        xx5=  self.avgpool(x5)  
        #         
        return x1,xx1,x2,x3,x4,x5

class AL_NET_decoder(nn.Module):#alccnet or no pretrain
    def __init__(self,img_ch=3, numclass=1):
        super(AL_NET_decoder, self).__init__()
        # self.resnet = Model() 
        n1 = 64 #28
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]
        # model_dict = torch.load('/home/zhaojing/AL-Net/weights/AL_NET_simsiam_encoder_jiarun1.th')
        # for k in list(model_dict.keys()):
        #     new_k=k.replace("module.","")
        #     model_dict[new_k]=model_dict.pop(k)
        #     # 载入参数
        # self.resnet.load_state_dict(model_dict)
        # self.resnet=self.resnet.backbone
        # self.resnet=AL_NET_encoder(img_ch=3, numclass=1)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.up= nn.Upsample(scale_factor=2)
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[2])
        self.att5=attentionLearn(s=filters[2])        
        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[2])
      
        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[1])
        self.att4=attentionLearn(s=filters[1])
       
        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[1])
    
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[0])
        self.att3=attentionLearn(s=filters[0])
        
        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[0])
    
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.att2=attentionLearn(s=filters[0])
       
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        
        self.Up1 = up_conv(ch_in=filters[0],ch_out=filters[0])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Conv3_1x1 = nn.Conv2d(filters[2],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv2_1x1 = nn.Conv2d(filters[1],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv1_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv0_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        # for param in self.parameters():
        #     param.requires_grad = False
        # unfreeze(self.encoder)
    def forward(self, x1,xx1,x2,x3,x4,x5):
        # encoding path
        # x,xk=torch.split(x,3,1)
        # _,_,_,_,_,_,e418=self.resnet(xk)
        # x1,xx1,x2,x3,x4,x5=self.resnet(x)
    
     
        # x55=  self.avgpool(torch.sigmoid(x5))
        # x418=  self.avgpool(torch.sigmoid(e418))
        # d=self.model(xx5,e418)
        
        # x5=self.sa512(x5,x5)
        # x5=self.sc512(x5)
        d5 = self.Up5(x5)
        xx4,at5=self.att5(x4)       
        d5 = torch.cat((xx4,d5),dim=1) 
        # d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)  
        
        d4 = self.Up4(d5)
        xx3,at4=self.att4(x3)
        d4 = torch.cat((xx3,d4),dim=1)
        # d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        xx2,at3=self.att3(x2)
        d3 = torch.cat((xx2,d3),dim=1)
        # d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1,at2=self.att2(xx1)
        d2 = torch.cat((x1,d2),dim=1)
        # d2 = torch.cat((xx1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d11 = self.Up1(d2)
        # att=self.Conv0_1x1(self.up(xx1))+self.Conv1_1x1(self.up(self.up(x2)))+self.Conv2_1x1(self.up(self.up(self.up(x3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(x4)))))
        att=self.Conv0_1x1(self.up(x1))+self.Conv1_1x1(self.up(self.up(xx2)))+self.Conv2_1x1(self.up(self.up(self.up(xx3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(xx4)))))
        d1 = self.Conv_1x1(d11)
        xx5=  self.avgpool(d11)
        return torch.sigmoid(d1),torch.sigmoid(att),xx5#,d
class AL_NET(nn.Module):#alccnet or no pretrain
    def __init__(self,img_ch=3, numclass=1):
        super(AL_NET, self).__init__()
         
        n1 = 64 #28
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4 
        self.conCode1=contextCode(channel=filters[0])
        self.conCode2=contextCode(channel=filters[0])
        self.conCode3=contextCode(channel=filters[1])
        self.conCode4=contextCode(channel=filters[2])
        
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.up= nn.Upsample(scale_factor=2)
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[2])
        self.att5=attentionLearn(s=filters[2])        
        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[2])
    
        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[1])
        self.att4=attentionLearn(s=filters[1])
       
        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[1])
    
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[0])
        self.att3=attentionLearn(s=filters[0])
        
        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[0])
    
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.att2=attentionLearn(s=filters[0])
       
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        
        self.Up1 = up_conv(ch_in=filters[0],ch_out=filters[0])
        
        self.Conv3_1x1 = nn.Conv2d(filters[2],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv2_1x1 = nn.Conv2d(filters[1],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv1_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv0_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
       
        for param in self.parameters():
            param.requires_grad = False 
        self.Conv_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
    def forward(self, x):
        # encoding path
        x = self.firstconv(x)
        x = self.firstbn(x)
        xx1 = self.firstrelu(x)
        x1 = self.firstmaxpool(xx1)
        x1=self.conCode1(x1)
        x2 = self.encoder1(x1)
        x2=self.conCode2(x2)
        x3 = self.encoder2(x2)
        x3=self.conCode3(x3)
        x4 = self.encoder3(x3)
        x4=self.conCode4(x4)
        x5 = self.encoder4(x4)        
 
        d5 = self.Up5(x5)
        xx4,at5=self.att5(x4)       
        d5 = torch.cat((xx4,d5),dim=1) 
        # d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)  
        
        d4 = self.Up4(d5)
        xx3,at4=self.att4(x3)
        d4 = torch.cat((xx3,d4),dim=1)
        # d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        xx2,at3=self.att3(x2)
        d3 = torch.cat((xx2,d3),dim=1)
        # d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1,at2=self.att2(xx1)
        d2 = torch.cat((x1,d2),dim=1)
        # d2 = torch.cat((xx1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        # att=self.Conv0_1x1(self.up(xx1))+self.Conv1_1x1(self.up(self.up(x2)))+self.Conv2_1x1(self.up(self.up(self.up(x3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(x4)))))
        att=self.Conv0_1x1(self.up(x1))+self.Conv1_1x1(self.up(self.up(xx2)))+self.Conv2_1x1(self.up(self.up(self.up(xx3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(xx4)))))
        d1 = self.Conv_1x1(d1)
        
        return torch.sigmoid(d1),torch.sigmoid(att)#,at2,at3,at4,at5

   

class attContextSeg_pretrain_noAtt(nn.Module):#alcc no attlearn or only context
    def __init__(self,img_ch=3, numclass=1):
        super(attContextSeg_pretrain_noAtt, self).__init__()
         
        n1 = 64 #28
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]
        resnet = models.resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4 
        self.conCode1=contextCode(channel=filters[0])
        self.conCode2=contextCode(channel=filters[0])
        self.conCode3=contextCode(channel=filters[1])
        self.conCode4=contextCode(channel=filters[2])
 
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.up= nn.Upsample(scale_factor=2)
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[2])
        #self.att5=attentionLearn(s=filters[2])        
        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[2])
    
        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[1])
        
        #self.att4=attentionLearn(s=filters[1])
       
        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[1])
    
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[0])
        
        #self.att3=attentionLearn(s=filters[0])
        
        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[0])
    
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        #self.att2=attentionLearn(s=filters[0])
       
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        
        self.Up1 = up_conv(ch_in=filters[0],ch_out=filters[0])
         
       
        self.Conv_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # encoding path
        x = self.firstconv(x)
        x = self.firstbn(x)
        xx1 = self.firstrelu(x)
        x1 = self.firstmaxpool(xx1)
        x1=self.conCode1(x1)
        x2 = self.encoder1(x1)
        x2=self.conCode2(x2)
        x3 = self.encoder2(x2)
        x3=self.conCode3(x3)
        x4 = self.encoder3(x3)
        x4=self.conCode4(x4)
        x5 = self.encoder4(x4)        
 
        d5 = self.Up5(x5)
        #xx4=self.att5(x4)       
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)  
        
        d4 = self.Up4(d5)
        #xx3=self.att4(x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        
        #xx2=self.att3(x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        
        #x1=self.att2(xx1)
        '''print(x1.shape[0])
        print(x1.shape[1])
        print(x1.shape[2])
        print(x1.shape[3])'''
        d2 = torch.cat((xx1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        
        #att=self.Conv0_1x1(self.up(x1))+self.Conv1_1x1(self.up(self.up(xx2)))+self.Conv2_1x1(self.up(self.up(self.up(xx3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(xx4)))))
        d1 = self.Conv_1x1(d1)
        
        return torch.sigmoid(d1)#,torch.sigmoid(att)
class AL_NET_without_Attention(nn.Module):#only pretrain or backbone
    def __init__(self,img_ch=3, numclass=1):
        super(AL_NET_without_Attention, self).__init__()
         
        n1 = 64 #28
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4 
        self.conCode1=contextCode(channel=filters[0])
        self.conCode2=contextCode(channel=filters[0])
        self.conCode3=contextCode(channel=filters[1])
        self.conCode4=contextCode(channel=filters[2])
 
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
      
        self.up= nn.Upsample(scale_factor=2)
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[2])
        self.att5=attentionLearn(s=filters[2])        
        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[2])
    
        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[1])
        self.att4=attentionLearn(s=filters[1])
       
        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[1])
    
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[0])
        self.att3=attentionLearn(s=filters[0])
        
        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[0])
    
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.att2=attentionLearn(s=filters[0])
       
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        
        self.Up1 = up_conv(ch_in=filters[0],ch_out=filters[0])
         
        self.Conv3_1x1 = nn.Conv2d(filters[2],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv2_1x1 = nn.Conv2d(filters[1],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv1_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv0_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        xx1 = self.firstrelu(x)
        x1 = self.firstmaxpool(xx1)
        x1=self.conCode1(x1)
        x2 = self.encoder1(x1)
        x2=self.conCode2(x2)
        x3 = self.encoder2(x2)
        x3=self.conCode3(x3)
        x4 = self.encoder3(x3)
        x4=self.conCode4(x4)
        x5 = self.encoder4(x4)        
 
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)  
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((xx1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        
        d1 = self.Up1(d2)
        d1 = self.Conv_1x1(d1)
        
        return torch.sigmoid(d1)

class AL_NET_without_Context(nn.Module):#alcc no context code or only attention learn
    def __init__(self,img_ch=3, numclass=1):
        super(AL_NET_without_Context, self).__init__()
         
        n1 = 64 #28
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 8]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4 
        self.conCode1=contextCode(channel=filters[0])
        self.conCode2=contextCode(channel=filters[0])
        self.conCode3=contextCode(channel=filters[1])
        self.conCode4=contextCode(channel=filters[2])
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
           
        self.up= nn.Upsample(scale_factor=2)
        self.Up5 = up_conv(ch_in=filters[3],ch_out=filters[2])
        self.att5=attentionLearn(s=filters[2])        
        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[2])
    
        self.Up4 = up_conv(ch_in=filters[2],ch_out=filters[1])
        self.att4=attentionLearn(s=filters[1])
       
        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[1])
    
        self.Up3 = up_conv(ch_in=filters[1],ch_out=filters[0])
        self.att3=attentionLearn(s=filters[0])
        
        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[0])
    
        self.Up2 = up_conv(ch_in=filters[0],ch_out=filters[0])
        self.att2=attentionLearn(s=filters[0])
       
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])
        
        self.Up1 = up_conv(ch_in=filters[0],ch_out=filters[0])
         
        self.Conv3_1x1 = nn.Conv2d(filters[2],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv2_1x1 = nn.Conv2d(filters[1],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv1_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv0_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)
        self.Conv_1x1 = nn.Conv2d(filters[0],numclass,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        # encoding path
        x = self.firstconv(x)
        x = self.firstbn(x)
        xx1 = self.firstrelu(x)
        x1 = self.firstmaxpool(xx1)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)           
    
        
        d5 = self.Up5(x5)
        xx4,at5=self.att5(x4)       
        d5 = torch.cat((xx4,d5),dim=1)        
        d5 = self.Up_conv5(d5)  
        
        d4 = self.Up4(d5)
        xx3,at4=self.att4(x3)
        d4 = torch.cat((xx3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        xx2,at3=self.att3(x2)
        d3 = torch.cat((xx2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1,at2=self.att2(xx1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Up1(d2)
        att=self.Conv0_1x1(self.up(x1))+self.Conv1_1x1(self.up(self.up(xx2)))+self.Conv2_1x1(self.up(self.up(self.up(xx3))))+self.Conv3_1x1(self.up(self.up(self.up(self.up(xx4)))))
        d1 = self.Conv_1x1(d1)
        
        return torch.sigmoid(d1),torch.sigmoid(att)
