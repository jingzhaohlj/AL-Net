import torch
from torch.functional import Tensor
from torchvision.transforms.transforms import CenterCrop
# from evaluation import *
from networks.cenet import *
from networks.alnet import *
from networks.attunet import AttU_Net
from networks.u2net import *
from networks.nestednet import *
from networks.nucleiSegnet import NucleiSegNet
from networks.joinseg import ResUNet34
import utils.surface_distance as surfdist
from framework import MyFrame
from loss import dice_bce_loss
from Constants import *
import AJI
import cv2
import os
import random
import numpy as np
from skimage import measure
from torchvision import transforms as T
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy.ndimage import  measurements
from skimage.segmentation import watershed
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
import numpy as np

def sortBylen(C):
    m = [] 
    n = []
    temp = []
    res = []
    for x in C:
        pts = np.asarray(x, dtype=np.int32)
        area = cv2.contourArea(pts)
        m.append(area)
        n.append(area)
    m.sort(reverse=True)
    index = 0
    for i in m:
        temp.append(C[n.index(i)])
        n[n.index(i)] = []
    for x in temp:
        index += 1
        if index <= 150:
            res.append(x)    
    return res

def postprocess_noring(sr,temp_image):
    sr[sr > 0.5] = 255
    sr[sr <= 0.5] = 0     
    image1 = T.ToPILImage()(sr[0].data.cpu()).convert('RGB')
    image = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2GRAY) 
    # _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = cv2.resize(image, (temp_image.shape[1],temp_image.shape[0])) 
    
    _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    predimage=image.copy()    
    dist_transform = cv2.distanceTransform(predimage,1,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    marker = measurements.label(sure_fg)[0]
    proced_pred = watershed(predimage, markers=marker, mask=image)
    # proced_pred, pred_num = measure.label(sr[0].detach().cpu().numpy(),connectivity = 1, return_num=True)
    return proced_pred,predimage

def postprocess(sr,temp_image,ring):
    sr[sr > 0.5] = 255#
    sr[sr <= 0.5] = 0     #
    image1 = T.ToPILImage()(sr[0].data.cpu()).convert('RGB')
    image = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2GRAY) 
    image = cv2.resize(image, (temp_image.shape[1],temp_image.shape[0])) 
    imagering = T.ToPILImage()(ring[0].data.cpu()).convert('RGB')
    imagering = cv2.cvtColor(np.asarray(imagering),cv2.COLOR_RGB2GRAY)  
    imagering=cv2.normalize(imagering,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    imagering = cv2.resize(imagering, (temp_image.shape[1],temp_image.shape[0])) 
    _,imagering=cv2.threshold(imagering,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    predimage=image.copy()  
    imagering=cv2.bitwise_and(imagering,predimage)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) 
    imagering = cv2.erode(imagering, kernel,iterations=1)
    imagering = cv2.dilate(imagering,  kernel,iterations=1)
    contourpoint, hierarchy= cv2.findContours ( imagering , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    imagep = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    sumArea=[]
    for ik in range(0,len(contourpoint)):
        sumArea.append(cv2.contourArea(contourpoint[ik]))
    arr_mean = np.mean(sumArea) 
    minarea=arr_mean/4
    for i in range(0,len(contourpoint)):
        if cv2.contourArea(contourpoint[i])>minarea:
            imagep = cv2.drawContours(imagep,contourpoint,i,255,-1)
    marker = measurements.label(imagep)[0]
    proced_pred = watershed(predimage, markers=marker, mask=image,watershed_line=True)
    contour=extractContour(proced_pred)
    
    return proced_pred,predimage

def drawcontour(temp_image,contours):
    imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    ring = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    imagei = np.zeros([temp_image.shape[0],temp_image.shape[1]], dtype=temp_image.dtype)
    contoursList=sortBylen(contours)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#定义结构元素的形状和大小
    for kk in range(0, len(contoursList)):
        imagei[imagei>0]=0
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        imagei = cv2.drawContours(imagei,pts,-1,255,-1)
        dst = cv2.dilate(imagei, kernel,iterations=2)#膨胀操作
        contours, hierarchy= cv2.findContours ( dst , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1 )
        imagemask = cv2.drawContours(imagemask,pts,-1,255,-1)
        imagemask = cv2.drawContours(imagemask,contours,-1,0,1)
        ring = cv2.drawContours(ring,pts,-1,255,1)

    ring = cv2.dilate(ring, kernel,iterations=1)#膨胀操作
    return imagemask,ring
def extractContour(proced_pred):
    contourS=[]
    proced=proced_pred.transpose(1,0)
    for ii in range(1,proced.max()+1):
        cont=[]
        cont1=[]
        b = np.where(proced == ii)
        x=0
       
        for a in range(0,len(b[0])):
            if x!=b[0][a]:
                cont.append([b[0][a],b[1][a]])
                if a!=0:
                    cont1.append([b[0][a],b[1][a-1]])
                x=b[0][a]
        cont1.reverse()
        cont = np.asarray([cont+cont1], dtype=np.int32)
        if len(cont[0])<6:
            continue
        contourS.append(cont)
    return contourS
def testNet(images_dir):
    size=448
    solver = MyFrame(AL_NET, dice_bce_loss, 3e-4)  
    solver.loadtar('/home/zhaojing/AL-Net/conweights/AL_NET_clusteredCell')
    testlist=['train-images']
    trainlist=['train-masks']
    ringlist=['train-ring']
    
    aji=0
    pq=0
    hd=0
    so=0
    ac=0
    JS=0
    DC=0
    index=0
    for ii in range(0,len(testlist)):
        images_name = os.listdir(images_dir+testlist[ii])
        images_name = sorted(images_name)  # 对文件按照文件名进行排序
        
        for image_name in tqdm(images_name):  # 对于每一个文件进行处理image 是文件的名字
            imgpath=os.path.join(images_dir+testlist[ii])
            temp_image = cv2.imread(os.path.join(imgpath, image_name)) 
            imagesize = cv2.resize(temp_image, (size,size))    
            maskpath=images_dir+trainlist[ii]
            ringpath=images_dir+ringlist[ii]
            if os.path.exists(maskpath)!=True:
                os.mkdir(maskpath)
            if os.path.exists(ringpath)!=True:
                os.mkdir(ringpath)
            maskname=image_name.replace("jpg","png")
            img = np.array(imagesize, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            img = torch.Tensor(img)
            img=img.unsqueeze(0)
            solver.set_input(img)#
            pred,ring = solver.test_image()# AL-Net
            # pred=solver.test_image_noring()
            sr=pred.clone()
            proced_pred,predimage=postprocess(sr,temp_image,ring)   # AL-Net 
            contour=extractContour(proced_pred)
            mask,ringimg=drawcontour(temp_image,contour)
            cv2.imwrite(os.path.join(maskpath, maskname),mask)
            cv2.imwrite(os.path.join(ringpath, maskname),ringimg)

if __name__ == '__main__':
    
    # testNet('/home/zhaojing/TargetA/')
    testNet('/home/zhaojing/TargetB/')

    