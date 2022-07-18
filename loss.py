import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):

        y_ce_true = y_true.squeeze(dim=1).long()


        a = self.ce_loss(y_pred, y_ce_true)

        return a


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b



def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))
class perceptual_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.names = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

    def forward(self, *args, **kwargs):
        x1_feat = args[0]
        x2_feat = args[1]

        loss = 0
        for key in self.names:
            size = x1_feat[key].size()
            # L1 loss
            # loss += (x1_feat[key] - x2_feat[key]).abs().sum() / (size[0] * size[1] * size[2] * size[3])
            # MSE loss
            loss += ((x1_feat[key] - x2_feat[key]) ** 2).sum() / (size[0] * size[1] * size[2] * size[3])

        loss /= 4
        return loss
class dice_shape_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_shape_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss(reduce=False)#
        self.celoss=nn.CrossEntropyLoss()

    def shape_loss(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        sloss=a.clone()
        shapeloss=torch.zeros_like(sloss)
        mask=torch.zeros_like(sloss)
        shapeloss=shapeloss.data.cpu()
        sr=y_pred.clone()
        pred=y_pred.clone()
        sr[sr > 0.5] = 1
        sr[sr <= 0.5] = 0 
        sr=sr+y_true
        rsize=5
        for k in range(0,len(pred)):
            predk=pred[k].int()
            srk=sr[k].int()          
            roisr=T.ToPILImage()(srk.data.cpu()).convert('L')
            image1 = T.ToPILImage()(predk.data.cpu()).convert('RGB')                   
            image = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2GRAY)  
            _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
            contours, hierarchy= cv2.findContours ( image , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )    
            for i in range(0,len(contours)):              
                xx, yy, ww, hh = cv2.boundingRect(contours[i])
                area=cv2.contourArea(contours[i])
                if area<100.0 or area>10000:
                    continue
                               
                roundness=(4*area)/(3.14*ww*hh)
                if roundness<0.3 or roundness>0.99:
                    continue
                j=0
                for a in range(0,len(contours[i])/rsize):              
                    
                    w=contours[i][j][0][0]
                    h= contours[i][j][0][1]
                    roi=np.asarray(roisr)                    
                    if w - rsize>0:
                        x = w - rsize
                    else:
                        x=0
                
                    if h - rsize>0:            
                        y = h - rsize
                    else:
                        y=0                
                
                    if h + rsize<448:
                        h = h+rsize
                    else:
                        h=448
                    if w+rsize<448:
                        w = w+rsize
                    else:
                        w=448
                    nt=0
                    ff=0
                    iou=0
                    index=[]
                    for b in range(x,w): 
                        for c in range(y,h):
                            if roi.item(c, b) == 2.0:
                                nt=nt+1                           
                            if roi.item(c, b) == 1.0:
                                ff=ff+1  
                                index.append([c,b])
                    if nt+ff !=0:
                        iou=nt/(nt+ff)
                        if iou>=0.99:
                            continue
                        va=((1-iou)*(1-roundness))
                        
                        for w in range(0,len(index)-1):
                            
                            shapeloss[k].index_fill_(1,torch.LongTensor(index[w]),va)
                          
                    j=j+rsize
        if torch.equal(shapeloss,mask.data.cpu())==True :
            
            return a.mean()
        else :
            lossShape=a*shapeloss.cuda()+a
           
            return lossShape.mean()
            
    def __call__(self, y_true, y_pred):
        
        b = self.shape_loss(y_true, y_pred)
        return b
def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1
    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(
        x, kernel_size=kernel_size,
        padding=padding,
        dilation=dilation
    )
    unfolded_x = unfolded_x.reshape(
        x.size(0), x.size(1), -1, x.size(2), x.size(3)
    )
    # remove the center pixels
    size = kernel_size ** 2
    unfolded_x = torch.cat((
        unfolded_x[:, :, :size // 2],
        unfolded_x[:, :, size // 2 + 1:]
    ), dim=2)
    return unfolded_x
class consistency_loss(nn.Module):
    def __init__(self, batch=True):
        super(consistency_loss, self).__init__()
        
        self.batch = batch
        self.bce_loss = nn.BCELoss(reduce=False)
    def consistency(self,y_true,  y_pred):
        a=self.bce_loss(y_pred, y_true)
        sloss=y_pred.clone().detach()
        thred=y_pred.clone().detach()
        unfolded_images = unfold_wo_center(
        sloss, kernel_size=3, dilation=2
        )            
        diff = abs(sloss[:, :, None] - unfolded_images)
        unfolded_weights = torch.max(diff, dim=2)[0]
        thred[thred<0.5]=0
        loss1=unfolded_weights*thred
        thred[thred>=0.5]=1
        loss=loss1+a
        return loss.mean()
    def consistency1(self,y_true,  y_pred,img):
        a=self.bce_loss(y_pred, y_true)
        sloss=y_pred.clone().detach()
        thred=y_pred.clone().detach()
        unfolded_images = unfold_wo_center(
        img, kernel_size=3, dilation=2
        )            
        diff = abs(img[:, :, None] - unfolded_images)
        unfolded_weights = torch.max(diff, dim=2)[0]
        thred[thred<0.5]=0
        loss1=unfolded_weights*thred
        thred[thred>=0.5]=1
        loss=loss1+a
        return loss.mean()

    def __call__(self, y_true, y_pred):
        
        b = self.consistency(y_true, y_pred)#,img,img
        return b
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a


import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i,:, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, target, input):
        target1 = torch.squeeze(target, dim=1)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target2 = target1.view(-1,1).long()

        logpt = F.log_softmax(input, dim=1)
        # print(logpt.size())
        # print(target2.size())
        logpt = logpt.gather(1,target2)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

