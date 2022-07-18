from dataclasses import replace
import torch
from torch.functional import Tensor
from torchvision.transforms.transforms import CenterCrop
from evaluation12 import *
from networks.cenet import *
from networks.unet import *
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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import numpy as np
import albumentations as A
def colortransforms(image):

    hue_shift = np.random.randint(10,50)
    sat_shift = np.random.randint(10,50)
    val_shift = np.random.randint(10,50)
  
    #hue_shift = np.uint8(hue_shift)
    transform = A.Compose([
        A.HueSaturationValue(hue_shift_limit=hue_shift, sat_shift_limit=sat_shift, val_shift_limit=val_shift, always_apply=False, p=0.5),
        A.RGBShift(r_shift_limit=hue_shift, g_shift_limit=sat_shift, b_shift_limit=val_shift, always_apply=False, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
        A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image
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
        if index <= 250:
            res.append(x)    
    return res

def postprocess_noring(sr,temp_image):
    # sr[sr > 0.5] = 255
    # sr[sr <= 0.5] = 0     
    image1 = T.ToPILImage()(sr[0].data.cpu()).convert('RGB')
    image = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2GRAY) 
    # _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = cv2.resize(image, (temp_image.shape[1],temp_image.shape[0])) 
    
    _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    predimage=image.copy()    
    dist_transform = cv2.distanceTransform(predimage,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    
    marker = measurements.label(sure_fg)[0]
    proced_pred = watershed(predimage, markers=marker, mask=image)
    # proced_pred, pred_num = measure.label(sr[0].detach().cpu().numpy(),connectivity = 1, return_num=True)
    return proced_pred,predimage

def postprocess(sr,temp_image,ring):
    # sr[sr > 0.7] = 255#
    # sr[sr <= 0.7] = 0     #
    image1 = T.ToPILImage()(sr[0].data.cpu()).convert('RGB')
    image = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2GRAY) 
    
    # _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = cv2.resize(image, (temp_image.shape[1],temp_image.shape[0])) 
    # cv2.imwrite('1.png',image)
    imagering = T.ToPILImage()(ring[0].data.cpu()).convert('RGB')
    imagering = cv2.cvtColor(np.asarray(imagering),cv2.COLOR_RGB2GRAY)  
    
    imagering=cv2.normalize(imagering,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    imagering = cv2.resize(imagering, (temp_image.shape[1],temp_image.shape[0])) 
    # cv2.imwrite('11.png',imagering)
    _,imagering=cv2.threshold(imagering,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    predimage=image.copy()
    # cv2.imwrite('2.png',predimage)
    # cv2.imwrite('22.png',imagering)    
    imagering=cv2.bitwise_and(imagering,predimage)
    # cv2.imwrite('3.png',imagering)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) 
    imagering = cv2.erode(imagering, kernel,iterations=1)
    imagering = cv2.dilate(imagering,  kernel,iterations=1)
    # cv2.imwrite('4.png',imagering)
    contourpoint, hierarchy= cv2.findContours ( imagering , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    imagep = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    sumArea=[]
    for ik in range(0,len(contourpoint)):
        sumArea.append(cv2.contourArea(contourpoint[ik]))
    arr_mean = np.mean(sumArea) 
    minarea=arr_mean/5
    for i in range(0,len(contourpoint)):
        if cv2.contourArea(contourpoint[i])>minarea:
            imagep = cv2.drawContours(imagep,contourpoint,i,255,-1)
    marker = measurements.label(imagep)[0]
    # cv2.imwrite('5.png',imagep)
    proced_pred = watershed(predimage, markers=marker, mask=image,watershed_line=True)
    return proced_pred,predimage
def genXMLMask(temp_image,maskpath):
    imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    mask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    tree = ET.parse(maskpath)
    root = tree.getroot()
    regions = root.findall('Annotation/Regions/Region')
    contoursList = []
    for region in regions:
        points = []
        for point in region.findall('Vertices/Vertex'):
            x = float(point.attrib['X'])
            y = float(point.attrib['Y'])
            if x>=224:
                x=223
            if y>=224:
                y=223
            points.append([x, y])
        if(len(points)<=0):
            continue
        pts = np.asarray([points], dtype=np.int32)
        contoursList.append(pts)
    contoursList=sortBylen(contoursList)
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        imagemask = cv2.drawContours(imagemask,pts,-1,kk+1,-1)
        mask = cv2.drawContours(mask,pts,-1,1,-1)
    return imagemask,mask
def genpredImage(predpath,temp_image):
    imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    mask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    with open(predpath, 'r') as rf:
            js = json.load(rf)
    shape = js['nuc']
    contoursList=[]
    for k in range(0, len(shape)):
        strk='%d'%(k+1)
        point = shape[strk]['contour']
        if point==None:
            return imagemask,mask
        points = []
        for j in range(0, len(point)):
            points.append([int(point[j][0]), int(point[j][1])])
        if len(points)<=0:
            continue
        pts = np.asarray([points], dtype=np.int32)
        contoursList.append(pts)
    contoursList=sortBylen(contoursList)
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        imagemask = cv2.drawContours(imagemask,pts,-1,kk+1,-1)
        mask = cv2.drawContours(mask,pts,-1,255,-1)
    return imagemask,mask
def genMask(temp_image,maskpath):
    # temp_image1=temp_image.copy()
    imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    mask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    with open(maskpath, 'r') as rf:
            js = json.load(rf)
    shape = js['shapes']
    contoursList=[]
    for k in range(0, len(shape)):
        point = shape[k]['points']
        points = []
        for j in range(0, len(point)):
            points.append([int(point[j][0]), int(point[j][1])])
        if len(points)<=0:
            continue
        pts = np.asarray([points], dtype=np.int32)
        contoursList.append(pts)
    # contoursList=sortBylen(contoursList)
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        r=random.randint(50,230)
        g=random.randint(30,150)
        b=random.randint(50,245)
        imagemask = cv2.drawContours(temp_image,pts,-1,(r,g,b),2)
    return imagemask

def computerFeature(temp_image,Contour):
    xx, yy, ww, hh = cv2.boundingRect(Contour)
    # area=cv2.contourArea(contours[i])
    # if area<100.0 or area>10000:
    #     continue
                    
    # roundness=(4*area)/(3.14*ww*hh)
    # if roundness<0.3 or roundness>0.99:
    #     continue

def grabCut1image(temp_image,pts):
    mask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    maskimg = np.full(temp_image.shape[:2], 2, dtype=np.uint8)
    mask = cv2.drawContours(mask,pts,-1,255,-1)
    # cv2.imwrite('/home/zhaojing/AL-Net/2.png',mask)mask.copy()#
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) 
    masktfg = cv2.erode(mask,kernel,iterations=1)
    masktbg = cv2.dilate(mask,kernel,iterations=7)
    contourfg, _= cv2.findContours ( masktfg , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    contourbg, _= cv2.findContours ( masktbg , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    maskimg = cv2.drawContours(maskimg,contourfg,-1,1,4)
    maskimg = cv2.drawContours(maskimg,contourbg,-1,0,5)
    # cv2.imwrite('mask',mask)
    # mask[mask > 0] = cv2.GC_PR_FGD
    # mask[mask == 0] = cv2.GC_BGD
    xx, yy, ww, hh = cv2.boundingRect(contourbg[0])
    rect =(xx, yy, ww, hh)# (1, 1, temp_image.shape[1],temp_image.shape[0])
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # 使用近似掩模分段在图像上执行Grabcut算法
    (maskimg, bgModel, fgModel) = cv2.grabCut(temp_image, maskimg, rect, bgModel,
                                        fgModel, iterCount=40, mode=cv2.GC_INIT_WITH_MASK)
    values = (
    ("Definite Background", cv2.GC_BGD),
    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
    )
    outputMask =  np.where((maskimg == 2) | (maskimg == 0)| (maskimg == 3), 0, 255).astype('uint8') 
    contourpoint, _= cv2.findContours ( outputMask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    return contourpoint

def drawcontour(temp_image,contours):
    temp_image1=temp_image.copy()
    mask_pred = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    contoursList=sortBylen(contours)
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        r=random.randint(50,230)
        g=random.randint(30,150)
        b=random.randint(50,245)
        mask_pred = cv2.drawContours(temp_image1,pts,-1,(r,g,b),2)
    return mask_pred
def extractContour(proced_pred):
    contourS=[]
    proced=proced_pred.transpose(1,0)
    for ii in range(1,proced.max()+1):
        cont=[]
        cont1=[]
        b = np.where(proced == ii)
        x=0
        if len(b[0])<5:
            continue
        for a in range(0,len(b[0])):
            if x!=b[0][a]:
                cont.append([b[0][a],b[1][a]])
                if a!=0:
                    cont1.append([b[0][a],b[1][a-1]])
                x=b[0][a]
        cont1.reverse()
        cont = np.asarray([cont+cont1], dtype=np.int32)
        contourS.append(cont)
    return contourS
def testNet(images_dir):
    size=448
    solver = MyFrame(AL_NET, dice_bce_loss, 3e-4)  #AL-NET_2e-4PatchSeg
    solver.loadtar('/home/zhaojing/AL-Net/patchweights/AL-NET_PatchSeg')#cenet\AL-Net_no_alloss_clusteredCell1
    images_name = os.listdir(images_dir)
    images_name = sorted(images_name)  # 对文件按照文件名进行排序
    
    for image_name in tqdm(images_name):  # 对于每一个文件进行处理image 是文件的名字
        imgpath=os.path.join(images_dir)
        temp_image = cv2.imread(os.path.join(imgpath, image_name)) 
        imagesize = cv2.resize(temp_image, (size,size))    
        maskname=image_name.replace("jpg","json")
        # maskname=image_name.replace("png","json")
        img = np.array(imagesize, np.float32).transpose(2, 0, 1) /255.0#* 3.2 - 1.6
        img = torch.Tensor(img)
        img=img.unsqueeze(0)
        solver.set_input(img)#
        pred,ring = solver.test_image()# AL-Net
   
        # pred=solver.test_image_noring()
        sr=pred.clone()
        proced_pred,predimage=postprocess(sr,temp_image,ring)   # AL-Net 
        # proced_pred,predimage=postprocess_noring(sr,temp_image)
        contour=extractContour(proced_pred)
        mask_pred=drawcontour(temp_image,contour)
        mask_true=genMask(temp_image,os.path.join(imgpath.replace('images','labels'), maskname))
        
        cv2.imwrite("/home/zhaojing/PatchSeg/result/mask_"+image_name,mask_true)
        cv2.imwrite("/home/zhaojing/PatchSeg/result/pred_"+image_name,mask_pred)
    

           
           



if __name__ == '__main__':
    
    # testNet('/home/zhaojing/clusteredCell/test1-images')
    testNet('/home/zhaojing/PatchSeg/test1-images')
    # testNet('/home/zhaojing/TargetA/test_images')
    # testNet('/home/zhaojing/TargetB/test-images/')
    # testNet('/repository01/nucleus_seg_data/MoNuSeg/test_224/')
    # testNet('/home/zhaojing/AL-Net/dataset/jiarun1/')
    