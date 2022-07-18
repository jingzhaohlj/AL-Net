# 用于将voc的分割数据转换为coco的标注格式，网上基本都是目标检测的转换，我门这里做的是分割的转换

import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from skimage import measure
import xml.etree.ElementTree as ET
START_BOUNDING_BOX_ID = 1  # ann开始的id
Image_ID=1      #image_id 开始的id


def read_cell_datasets(root_path, mode,datasetname):
    
    if mode=="test":
        if datasetname=="patchSeg":
            list_name=['test-images']
        else :
            list_name=['test-difficult','test-normal','test-sample']
    elif mode =="train":
        if datasetname=="patchSeg":
            list_name=['train-images']
        else :
            list_name=['difficult','normal','sample']#
    for i in range(0,len(list_name)):
        image_root = os.path.join(root_path, list_name[i])
        gt_root = os.path.join(root_path, list_name[i].replace('images','labels'))
        for image_name in tqdm(os.listdir(image_root)):
            image_path = os.path.join(image_root, image_name)
            label_path = os.path.join(gt_root, image_name)
            filename = image_name.split('.')
            GT_name =filename[0] + '.json'  
            # GT_name =filename[0] + '.xml'      
            label_path = os.path.join(gt_root, GT_name)        
            data_ringuf_seg_loader(image_path, label_path)
            # data_ring_xml_seg_loader(image_path, label_path)

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
                
def data_ring_xml_seg_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    root_path=mask_path.split('_xml')
    name=root_path[1].split('.')
    ring_path=root_path[0]+'_ring'
    if os.path.exists(ring_path)==False:
        os.mkdir(ring_path)
    masks_path=root_path[0]+'_masks'
    if os.path.exists(masks_path)==False:
        os.mkdir(masks_path)
    imagemask = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    imagering = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    imagei = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    tree = ET.parse(mask_path)
    root = tree.getroot()
    regions = root.findall('Annotation/Regions/Region')
    allpoints = []
    for region in regions:
        points = []
        for point in region.findall('Vertices/Vertex'):
            x = float(point.attrib['X'])
            y = float(point.attrib['Y'])
            points.append([x, y])
        if(len(points)<=0):
            continue
        allpoints.append(points)
    allpoints=sortBylen(allpoints)
    for k in range(0, len(allpoints)):
        imagei[imagei>0]=0
        pts = np.asarray([allpoints[k]], dtype=np.int32)
        imagei = cv2.drawContours(imagei,pts,-1,255,-1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))#定义结构元素的形状和大小
        dst = cv2.dilate(imagei, kernel,iterations=2)#膨胀操作
        contours, hierarchy= cv2.findContours ( dst , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1 )
        imagemask = cv2.drawContours(imagemask,pts,-1,255,-1)
        imagemask = cv2.drawContours(imagemask,contours,-1,0,1)

    imagering = cv2.Canny(imagemask, 50, 150) 
    imagering = cv2.dilate(imagering, kernel,iterations=1)#膨胀操作
    cv2.imwrite(ring_path+name[0]+'.png',imagering)
    cv2.imwrite(masks_path+name[0]+'.png',imagemask)
def data_ringuf_seg_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    root_path=mask_path.split('-labels')
    name=root_path[1].split('.')
    ring_path=root_path[0]+'-ring'
    if os.path.exists(ring_path)==False:
        os.mkdir(ring_path)
    masks_path=root_path[0]+'-masks'
    if os.path.exists(masks_path)==False:
        os.mkdir(masks_path)
    imagemask = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    imagering = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    imagei = np.zeros([img.shape[0],img.shape[1]], dtype=img.dtype)
    with open(mask_path, 'r') as rf:
            js = json.load(rf)
    shape = js['shapes']
    allpoints = []
    for i in range(0, len(shape)):
        point = shape[i]['points']
        points = []        
        for j in range(0, len(point)):
            points.append([int(point[j][0]), int(point[j][1])])
        if len(points)<=0:
            continue
        allpoints.append(points)
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
        imagering = cv2.drawContours(imagering,pts,-1,255,1)

    imagering = cv2.dilate(imagering, kernel,iterations=1)#膨胀操作
    cv2.imwrite(ring_path+name[0]+'.png',imagering)
    cv2.imwrite(masks_path+name[0]+'.png',imagemask)


if __name__ == "__main__":
   
    root_path='/home/zhaojing/PatchSeg'
    read_cell_datasets(root_path,'test','patchSeg')

