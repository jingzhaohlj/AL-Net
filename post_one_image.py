from scipy.sparse.construct import random
import torch
from evaluation import *
from networks.cenet import CE_Net_
import random
from networks.alnet import *
from networks.joinseg import *
from networks.cenet import *
from networks.attunet import *
from networks.u2net import *
from networks.nestednet import *
from networks.nucleiSegnet import *
from networks.unet import *
from framework import MyFrame
from loss import dice_bce_loss
from Constants import *
from skimage import measure
import AJI
import cv2
import os
import numpy as np
from torchvision import transforms as T

from tqdm import tqdm
from scipy.ndimage import  measurements
from skimage.segmentation import watershed
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
   
    marker = measurements.label(sure_fg)[0]
    proced_pred = watershed(predimage, markers=marker, mask=image,watershed_line=True)
    
    # proced_pred, pred_num = measure.label(predimage,connectivity = 1, return_num=True)
    return proced_pred,predimage

def postprocess(sr,temp_image,ring):
    sr[sr > 0.5] = 255
    sr[sr <= 0.5] = 0     
    image1 = T.ToPILImage()(sr[0].data.cpu()).convert('RGB')
    image = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2GRAY) 
    # _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = cv2.resize(image, (temp_image.shape[1],temp_image.shape[0])) 
    imagering = T.ToPILImage()(ring[0].data.cpu()).convert('RGB')
    imagering = cv2.cvtColor(np.asarray(imagering),cv2.COLOR_RGB2GRAY)  
    imagering=cv2.normalize(imagering,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    _,imagering=cv2.threshold(imagering,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    imagering = cv2.resize(imagering, (temp_image.shape[1],temp_image.shape[0])) 
    _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    predimage=image.copy()    
  
    imagering=cv2.bitwise_and(imagering,predimage)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
    imagering = cv2.erode(imagering, kernel,iterations=1)
    imagering = cv2.dilate(imagering,  kernel,iterations=2)

  
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
    proced_pred = watershed(predimage, markers=marker, mask=image,compactness=5)
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
    contoursList=sortBylen(contoursList)
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        imagemask = cv2.drawContours(imagemask,pts,-1,kk+1,-1)
        mask = cv2.drawContours(mask,pts,-1,1,-1)
    return imagemask,mask

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
def toMask(maskpath,predcontour):
    temp_image = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE) 
    imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    mask_pred = np.zeros([temp_image.shape[0], temp_image.shape[1],3], dtype=temp_image.dtype)
    mask=temp_image.copy()
    contours, hierarchy= cv2.findContours ( mask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1 )
    contoursList=sortBylen(contours)
    for k in range(0, len(contoursList)):
        pts = np.asarray([contoursList[k]], dtype=np.int32)
        imagemask = cv2.drawContours(imagemask,pts,-1,k+1,-1)
        r=random.randint(50,230)
        g=random.randint(30,150)
        b=random.randint(50,245)
        mask_pred = cv2.drawContours(mask_pred,pts,-1,(r,g,b),-1)
    mask_pred = cv2.drawContours(mask_pred,predcontour,-1,(0,255,0),1)
    return imagemask,mask_pred,contoursList
def genMask(maskpath,temp_image,predcontour):
    imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    mask_pred = np.zeros([temp_image.shape[0], temp_image.shape[1],3], dtype=temp_image.dtype)
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
    for k in range(0, len(contoursList)):
        pts = np.asarray([contoursList[k]], dtype=np.int32)
        imagemask = cv2.drawContours(imagemask,pts,-1,k+1,-1)
        r=random.randint(50,230)
        g=random.randint(30,150)
        b=random.randint(50,245)
        mask_pred = cv2.drawContours(mask_pred,pts,-1,(r,g,b),-1)
    mask_pred = cv2.drawContours(mask_pred,predcontour,-1,(0,255,0),1)
    return imagemask,mask_pred,contoursList
def testNet(images_dir):
    
    solver = MyFrame(AttU_Net, dice_bce_loss, 2e-4)  
    solver.loadtar('/home/zhaojing/AL-Net/weights/AttU_NetBNS')#cenet\
    images_name = os.listdir(images_dir)
    images_name = sorted(images_name)  # 对文件按照文件名进行排序
    i=0
    size=448
    for image_name in tqdm(images_name):  # 对于每一个文件进行处理image 是文件的名字
        temp_image = cv2.imread(os.path.join(images_dir, image_name)) 
        imageGray = cv2.cvtColor(temp_image,cv2.COLOR_RGB2GRAY)  
        name=image_name.split('.') 
        imagesize = cv2.resize(temp_image, (size,size)) 
        # maskpath=images_dir.replace("testimage","testimage_json")#
        # maskpath=images_dir.replace("test1-images","test1-labels")#EDF
        maskpath=images_dir.replace("test1_images","test1_json")
        # maskname=image_name.replace("jpg","png")
        maskname=image_name.replace("jpg","json")
        img = np.array(imagesize, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
        img = torch.Tensor(img)
        img=img.unsqueeze(0)
        solver.set_input(img)#
        # pred,ring = solver.test_image()#
        pred=solver.test_image_noring()
        sr=pred.clone()
        # proced_pred,predimage=postprocess(sr,temp_image,ring)    
        proced_pred,predimage=postprocess_noring(sr,temp_image)
        # proced_pred,predimage=genpredImage('/home/zhaojing/hover_net/output/epoch46/json/'+maskname,temp_image)
        contourS=extractContour(proced_pred)
        maskimg=temp_image.copy()
        img_s = cv2.drawContours(temp_image,contourS,-1,(0,255,0),2)
        # imagemask,mask_pred,contoursList=toMask(maskpath+maskname,contourS)
        imagemask,mask_pred,contoursList=genMask(maskpath+maskname,temp_image,contourS)
        mask_img = cv2.drawContours(maskimg,contoursList,-1,(0,255,0),2)
        aji1 = AJI.get_fast_aji(imagemask,proced_pred)
        pq1=AJI.get_fast_pq(np.array(imagemask,dtype='uint8'),np.array(proced_pred,dtype='uint8'))
        str=name[0]+"_aji:%.4f"%aji1+"_pq:%.4f"%pq1                
        cv2.imwrite('/home/zhaojing/AL-Net/dataset/result/'+str+'-image.png', img_s) 
        cv2.imwrite('/home/zhaojing/AL-Net/dataset/result/'+str+'-pred.png', mask_pred) 
        cv2.imwrite('/home/zhaojing/AL-Net/dataset/result/'+str+'-mask.png', mask_img) 
 


if __name__ == '__main__':
    
    # testNet('/home/zhaojing/dataset/clusteredCell/testimage/')
    # testNet('/home/zhaojing/ALCC-net/dataset/EDF/test1-images/')
    testNet('/repository01/nucleus_seg_data/BNS/test1_images/')