from albumentations.augmentations.geometric.transforms import ElasticTransform
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from AutomaticWeightedLoss import AutomaticWeightedLoss
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from torchvision import transforms as T
# from pyheatmap.heatmap import HeatMap
from torchvision import models
from loss import *
from networks.alnet import *
class vgg16_feat(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        # self.ennet = AL_NET_encoder().cuda()
        # self.ennet = torch.nn.DataParallel(self.ennet, device_ids=range(torch.cuda.device_count()))
        # self.awl = AutomaticWeightedLoss(3).cuda()	# we have 2 losses
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.net.parameters()},
        #     {'params': self.awl.parameters(), 'weight_decay': 0}],lr=lr)lambda p: p.requires_grad,
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.enoptimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.ennet.parameters()), lr=lr)
        # self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)
        self.vgg_model = vgg16_feat().cuda()  
        self.vgg_model = nn.DataParallel(self.vgg_model,device_ids=range(torch.cuda.device_count()))  
        self.loss = loss()
        self.old_lr = lr
        self.bce = dice_bce_loss()
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        self.criterion_perceptual = perceptual_loss()
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
    def optimize_ref(self):
        self.net.train(True)
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img) 
        mask,maskr=torch.split(self.mask,1,1)         
        bceloss = self.loss(mask, pred)
        pred_map = pred.repeat(1,3,1,1).float()    # only care about the contours
        target_map = mask.repeat(1,3,1,1).float() 
        pred_feat = self.vgg_model(pred_map)
        target_feat = self.vgg_model(target_map)
        loss_perceptual = self.criterion_perceptual(pred_feat, target_feat)
        loss = bceloss +0.1 * loss_perceptual        
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.data,pred,mask 
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        mask,maskr=torch.split(self.mask,1,1)
        loss = self.loss(mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data, pred,mask
    def optimize_alnet(self):
        self.forward()
        self.optimizer.zero_grad()
        pred,ring = self.net.forward(self.img)
        mask,maskr=torch.split(self.mask,1,1)
        loss = self.loss(mask, pred)#,self.img
        lossr = self.bce(maskr, ring)
        # awl = AutomaticWeightedLoss(2),self.img
        # loss = awl(loss, lossr)
        loss=loss+lossr
        loss.backward()
        self.optimizer.step()
        
        return loss.data, pred,mask 
    def optimize_seg_class(self):
        self.net.train(True)
        self.ennet.train(True)
        self.forward()
        self.enoptimizer.zero_grad()
        self.optimizer.zero_grad()
        x,xk=torch.split(self.img,3,1)
        x1,xx1,x2,x3,x4,x5 = self.ennet.forward(x)
        pred,ring,_ = self.net.forward(x1,xx1,x2,x3,x4,x5)
        mask,maskr=torch.split(self.mask,1,1)
        loss = self.loss(mask, pred)
        lossr = self.loss(maskr, ring) 
        loss=loss+lossr                           
        loss.backward() #  bceloss.clone().detach() 
        self.enoptimizer.step()
        self.optimizer.step()
        return loss.data, pred,mask
    def test_seg_class(self):
        self.net.train(False)
        self.ennet.train(False)
        self.forward()
      
        x,xk=torch.split(self.img,3,1)
        with torch.no_grad():
            x1,xx1,x2,x3,x4,x5 = self.ennet.forward(x)
            pred,ring,_ = self.net.forward(x1,xx1,x2,x3,x4,x5)
        mask,maskr=torch.split(self.mask,1,1)
       
        return pred,mask
    def test_ende(self):
        self.net.train(False)
        self.ennet.train(False)
        self.forward()
      
        x,xk=torch.split(self.img,3,1)
        with torch.no_grad():
            x1,xx1,x2,x3,x4,x5 = self.ennet.forward(x)
            pred,ring,_ = self.net.forward(x1,xx1,x2,x3,x4,x5)
       
       
        return pred,ring
    def optimize_atloss(self):
        self.forward()
        self.optimizer.zero_grad()
        pred,ring,silloss = self.net.forward(self.img)
        # silloss=1-silloss
        mask,maskr=torch.split(self.mask,1,1)
        loss = self.loss(mask, pred)
        lossr = self.loss(maskr, ring)
        # awl = AutomaticWeightedLoss(2)
        # loss = awl(loss, lossr)
        loss=loss+silloss+lossr
        loss.backward()
        self.optimizer.step()
        return loss.data, pred,mask  
    def optimize_simsiam(self):
        self.forward()
        self.optimizer.zero_grad()
        x,xk=torch.split(self.img,3,1)
        d1,d2 = self.net.forward(x,xk)
        silloss=d1+d2
        silloss.backward()
        self.optimizer.step()
        return silloss.data   
    def test_alnet(self):
        self.net.train(False)
        self.forward()
        self.optimizer.zero_grad()
        with torch.no_grad():
            pred,ring = self.net.forward(self.img)
        mask,maskr=torch.split(self.mask,1,1)
    
        return pred,mask    
    def test_image(self):
        self.net.train(False)
        self.forward()
        self.optimizer.zero_grad()
        with torch.no_grad():
            pred,ring = self.net.forward(self.img)
        #mask,maskr=torch.split(self.mask,1,1)
    
        return pred,ring 
    def drawHeatmap(self,sns,at,name):  
        hm=at[0].data.cpu().numpy()
        mask = cv2.resize(hm, (448, 448))
        ax = sns.heatmap(mask, cmap='rainbow')
        # plt.savefig('/home/zhaojing/AL-Net/dataset/'+name+'.png', dpi=600) #指定分辨率保存
        
    def test_image_att(self):
        self.net.train(False)
        self.forward()
        self.optimizer.zero_grad()
        with torch.no_grad():
            pred,ring ,at2,at3= self.net.forward(self.img)
        #mask,maskr=torch.split(self.mask,1,1)
        sns.set()
        # self.drawHeatmap(sns,at2[0],'at2al')
        # self.drawHeatmap(sns,at3[0],'at3al')
        self.drawHeatmap(sns,at3[0],'at4al')
        self.drawHeatmap(sns,at2[0],'at5al')
        return pred,ring    
    def test_image_noring(self):
        self.net.train(False)
        self.forward()
        self.optimizer.zero_grad()
        with torch.no_grad():
            pred = self.net.forward(self.img)
        #mask,maskr=torch.split(self.mask,1,1)
    
        return pred   
    def test_cenet(self):
        self.net.train(False)
        self.forward()
        self.optimizer.zero_grad()
        with torch.no_grad():
            pred = self.net.forward(self.img)
        mask,maskr=torch.split(self.mask,1,1)
    
        return pred,mask 
    def save(self, path):
        torch.save(self.net.state_dict(), path)
    def savetar(self, path,epoch,epoch_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': epoch_loss,
            'lr':self.old_lr},path+'.tar')    
        # torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': self.ennet.state_dict(),
        #         'optimizer_state_dict': self.enoptimizer.state_dict(),
        #         'loss': epoch_loss,
        #         'lr':self.old_lr},path+'en.tar') 
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': self.net.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'loss': epoch_loss,
        #     'lr':self.old_lr},path+'de.tar') 
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    def loadtar(self,path):
        checkpoint = torch.load(path+'.tar')
        model_dict=checkpoint['model_state_dict']
        # self.optimizer./load_state_dict(checkpoint['optimizer_state_dict'])
        self.net.load_state_dict(model_dict)
        # print(self.optimizer.state_dict()['param_groups'][0]['lr'])

        # checkpoint = torch.load(path+'en.tar')

        # model_dict=checkpoint['model_state_dict']
        # # for k in list(model_dict.keys()):
        # #     new_k=k.replace("en","module.en")
        # #     new_k=new_k.replace("co","module.co")
        # #     new_k=new_k.replace("fi","module.fi")
        # #     model_dict[new_k]=model_dict.pop(k)
        # self.ennet.load_state_dict(model_dict)
        # checkpoint = torch.load(path+'de.tar')
        # model_dict=checkpoint['model_state_dict']
        # # for k in list(model_dict.keys()):
        # #     new_k=k.replace("module.","")
        # #     model_dict[new_k]=model_dict.pop(k)
        #     # 载入参数
        # self.net.load_state_dict(model_dict)
        self.epoch=checkpoint['epoch']
        self.old_lr=checkpoint['lr']
        # print(self.old_lr)
        # print (self.net)
        
    def update_lr(self, new_lr,  factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        # print (mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
