import os
import random
from random import shuffle
import numpy as np
import cv2
import base64
import json
import os.path as osp
import importlib
from PIL import Image
import nibabel as nib
import skimage.io as io
from nibabel.viewers import OrthoSlicer3D
import matplotlib.pyplot as plt
import imageio     

path='D:/zhaojing/labelData/label/BNS/'
imagesize=256

def genJson(path_open,vecResRects,labelName):    

    MeserList = []
    for i in range(len(vecResRects)):
        listTemp1 = []
        for j in range(len(vecResRects[i])):
            listTemp1.append(vecResRects[i][j][0].tolist())
        MeserList.append(listTemp1)   
    file = open(path_open, 'a')  # 追加的方式写入
    your_dict = {}
    your_dict['version'] = "3.15.2"
    your_dict['flags'] = {}
    shapes_list = []
    for i in range(0,len(MeserList)):
        your_dict_shape = {}
        your_dict_shape['label'] = labelName
        your_dict_shape['line_color'] = None
        your_dict_shape['fill_color'] = None
        your_dict_shape['points'] = MeserList[i]
        your_dict_shape['shape_type'] = "polygon"
        your_dict_shape['flags'] = {}
        shapes_list.append(your_dict_shape)
    your_dict['shapes'] = shapes_list
    json_str = json.dumps(your_dict,indent=2)  # 将字典装化为json串
    file.write(json_str + '\n')    
                                           

def extractContour(proced_pred):
    contourS=[]
    for ii in range(0,proced_pred.max()):
        imagei = np.zeros([imagesize,imagesize,1], dtype=np.uint8)
        imagei[proced_pred==ii+1]=255
        contours, hierarchy= cv2.findContours ( imagei , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1 )
        for i in  range(0,len(contours)):
            area = cv2.contourArea(contours[i])
            if area<100.0:                                      #设置目标的最小面积，可以不设置，但是我再转换的过程中发现很多area=0.0 的情况，所以最好还是设置一下
                continue 
            contourS.append(contours[i])
    if (len(contourS)<5):
        return None
    else:   
        return contourS 
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
def data_ringuf_seg_loader(mode,allpoints,img,num,name):
    root_path=path+mode
    img_path=root_path+'_images/'
    if os.path.exists(img_path)==False:
        os.mkdir(img_path)
    cv2.imwrite(img_path+name+'_%d.jpg'%(num),img)
    json_path=root_path+'_json/'
    if os.path.exists(json_path)==False:
        os.mkdir(json_path)
    genJson(json_path+name+'_%d.json'%(num),allpoints,'nuclei')
    ring_path=root_path+'_ring/'
    if os.path.exists(ring_path)==False:
        os.mkdir(ring_path)
    masks_path=root_path+'_masks/'
    if os.path.exists(masks_path)==False:
        os.mkdir(masks_path)
    imagemask = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    imagering = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    imagei = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    allpoints=sortBylen(allpoints)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#定义结构元素的形状和大小
    for k in range(0, len(allpoints)):
        imagei[imagei>0]=0
        pts = np.asarray([allpoints[k]], dtype=np.int32)
        imagei = cv2.drawContours(imagei,pts,-1,255,-1)
        dst = cv2.dilate(imagei, kernel,iterations=2)#膨胀操作
        contours, hierarchy= cv2.findContours ( dst , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1 )
        imagemask = cv2.drawContours(imagemask,pts,-1,255,-1)
        imagemask = cv2.drawContours(imagemask,contours,-1,0,1)

    imagering = cv2.Canny(imagemask, 50, 150) 
    imagering = cv2.dilate(imagering, kernel,iterations=1)#膨胀操作
    cv2.imwrite(ring_path+name+'_%d.png'%(num),imagering)
    cv2.imwrite(masks_path+name+'_%d.png'%(num),imagemask)

def enhance(roi,roisrc,num,mode,name):
    contourS1=extractContour(roi)
    if contourS1!=None:
        data_ringuf_seg_loader(mode,contourS1,roisrc,num+1,name)
    img_trans=cv2.transpose(roisrc)
    img_flip0=cv2.flip(roisrc,0)
    img_flip1=cv2.flip(roisrc,1)
    img_flip_1=cv2.flip(roisrc,-1) 
    mimg_trans=cv2.transpose(roi)
    mimg_flip0=cv2.flip(roi,0)
    mimg_flip1=cv2.flip(roi,1)
    mimg_flip_1=cv2.flip(roi,-1)
    contourS2=extractContour(mimg_trans)
    if contourS2!=None:
        data_ringuf_seg_loader(mode,contourS2,img_trans,num+2,name)
    contourS3=extractContour(mimg_flip0)
    if contourS3!=None:
        data_ringuf_seg_loader(mode,contourS3,img_flip0,num+3,name)
    contourS4=extractContour(mimg_flip1)
    if contourS4!=None:
        data_ringuf_seg_loader(mode,contourS4,img_flip1,num+4,name)
    contourS5=extractContour(mimg_flip_1)
    if contourS5!=None:
        data_ringuf_seg_loader(mode,contourS5,img_flip_1,num+5,name)


if __name__ == '__main__':
    #   print(torch.__version__())
    mode='test'
    i=0
 
    pathlist='D:/zhaojing/labelData/label/BNS/test-bns/'
    pathlabel='D:/zhaojing/labelData/label/BNS/test-bns-gt/'
    for filename in os.listdir(pathlist):
        i=i+1
        grayimage = cv2.imread(pathlist + filename)
        filename = filename.split('.')
        image_data = nib.load(pathlabel + filename[0]+'.nii.gz')

        img_fdata=image_data.dataobj
        new_data1 =np.array(img_fdata)
        
        new_data1 = cv2.flip(new_data1, 1)
        rows, cols = new_data1.shape
        rotate = cv2.getRotationMatrix2D((rows*0.5, cols*0.5), 90, 1)
        new_data1 = cv2.warpAffine(new_data1, rotate, (cols, rows))
        new_data=new_data1.copy()
        
        for j in range(0,2):
            for k in range(0,2):
                roi = new_data[j*256:j*256+imagesize, k*256:k*256+imagesize]
                roisrc=grayimage[j*256:j*256+imagesize, k*256:k*256+imagesize]
                num=i*1000+j*100+k*10
                enhance(roi, roisrc, num,mode,filename[0]) 
        for j in range(0,2):
            for k in range(0,2):        
                roi = new_data[j*224:j*224+300, k*224:k*224+300]
                roisrc=grayimage[j*224:j*224+300, k*224:k*224+300]
                roi = cv2.resize(roi, (imagesize, imagesize))
                roisrc = cv2.resize(roisrc, (imagesize, imagesize))
                num=(i+10)*1000+j*100+k*10
                enhance(roi, roisrc, num,mode,filename[0])     
        for j in range(0,2):
            for k in range(0,2):        
                roi = new_data[j*100:j*100+356, k*100:k*100+356]
                roisrc=grayimage[j*100:j*100+356, k*100:k*100+356]
                num=(i+20)*1000+j*100+k*10
                roi = cv2.resize(roi, (imagesize, imagesize))
                roisrc = cv2.resize(roisrc, (imagesize, imagesize))
                enhance(roi, roisrc, num,mode,filename[0])     
        for j in range(0,2):
            for k in range(0,2):
                roi = new_data[j*224:j*224+224, k*224:k*224+224]
                roisrc=grayimage[j*224:j*224+224, k*224:k*224+224]
                num=(i+30)*1000+j*100+k*10
                roi = cv2.resize(roi, (imagesize, imagesize))
                roisrc = cv2.resize(roisrc, (imagesize, imagesize))                
                enhance(roi, roisrc, num,mode,filename[0])       

        

      