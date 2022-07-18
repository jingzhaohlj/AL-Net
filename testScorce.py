import torch
from torch.functional import Tensor
from torchvision.transforms.transforms import CenterCrop
from Evaluation import *
from networks.cenet import *
from networks.alnet import *
from networks.attunet import AttU_Net
from networks.u2net import *
from networks.nestednet import *
from networks.unet import *
from networks.nucleiSegnet import NucleiSegNet
from networks.joinseg import ResUNet34
import utils.surface_distance as surfdist
from framework import MyFrame
from loss import dice_bce_loss
from Constants import *
# import AJI
import cv2
import os
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
        if index <= 150:
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
    # sr[sr > 0.4] = 255#
    # sr[sr <= 0.4] = 0     #
    image1 = T.ToPILImage()(sr[0].data.cpu()).convert('RGB')
    image = cv2.cvtColor(np.asarray(image1),cv2.COLOR_RGB2GRAY) 
    _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # cv2.imwrite('1.png',image)
    image = cv2.resize(image, (temp_image.shape[1],temp_image.shape[0])) 
    imagering = T.ToPILImage()(ring[0].data.cpu()).convert('RGB')
    imagering = cv2.cvtColor(np.asarray(imagering),cv2.COLOR_RGB2GRAY)  
    imagering=cv2.normalize(imagering,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)
    _,imagering=cv2.threshold(imagering,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imwrite('1.png',imagering)
    imagering1 = cv2.resize(imagering, (temp_image.shape[1],temp_image.shape[0])) 
    _,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    predimage=image.copy()    
    imagering=cv2.bitwise_and(imagering1,predimage)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3, 3)) 
    imagering = cv2.erode(imagering, kernel,iterations=1)
    imagering = cv2.dilate(imagering,  kernel,iterations=1)
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
    true = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    
    with open(maskpath, 'r') as rf:
            js = json.load(rf)
    shape = js['shapes']
    contoursList=[]
    for k in range(0, len(shape)):
        point = shape[k]['points']
        points = []
        for j in range(0, len(point)):
            points.append([int(point[j][0]), int(point[j][1])])
        if len(points)<=5:
            continue
        pts = np.asarray([points], dtype=np.int32)
        contoursList.append(pts)
    contoursList=sortBylen(contoursList)
    true_instance=[None,]
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
        imagemask = cv2.drawContours(imagemask,pts,-1,1,-1)
        true = cv2.drawContours(true,pts,-1,kk+1,-1)
        true_instance.append(imagemask)
    return true_instance,true
def toMask(temp_image,maskpath):
    imagemask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    mask = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
    temp_image = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE) 
    contours, hierarchy= cv2.findContours ( temp_image , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1 )
    contoursList=sortBylen(contours)
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        imagemask = cv2.drawContours(imagemask,pts,-1,kk+1,-1)
        mask = cv2.drawContours(mask,pts,-1,1,-1)
    return imagemask,mask
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
def grabCut1image(temp_image,mask):
    maskimg = np.full(temp_image.shape[:2], 2, dtype=np.uint8)
   
    # cv2.imwrite('/home/zhaojing/AL-Net/2.png',mask)mask.copy()#
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) 
    masktfg = mask.copy()#cv2.erode(mask,kernel,iterations=1)
    masktbg = cv2.dilate(mask,kernel,iterations=3)
    contourfg, _= cv2.findContours ( masktfg , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    contourbg, _= cv2.findContours ( masktbg , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    maskimg = cv2.drawContours(maskimg,contourfg,-1,1,-1)
    maskimg = cv2.drawContours(maskimg,contourbg,-1,0,2)
    # cv2.imwrite('mask',mask)
    # mask[mask > 0] = cv2.GC_PR_FGD
    # mask[mask == 0] = cv2.GC_BGD
    rect = (1, 1, temp_image.shape[1],temp_image.shape[0])
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # 使用近似掩模分段在图像上执行Grabcut算法
    (maskimg, bgModel, fgModel) = cv2.grabCut(temp_image, maskimg, rect, bgModel,
                                        fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)
    values = (
    ("Definite Background", cv2.GC_BGD),

    ("Probable Background", cv2.GC_PR_BGD),
    ("Definite Foreground", cv2.GC_FGD),
    ("Probable Foreground", cv2.GC_PR_FGD),
    )
    outputMask =  np.where((maskimg == 2) | (maskimg == 0), 0, 255).astype('uint8') 
    # contourpoint, _= cv2.findContours ( outputMask , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    return outputMask
def ConvertCoordinates(image,contour,direction):
    padding=10
    xx, yy, ww, hh=cv2.boundingRect(contour)
    contour[0]=contour[0]-direction*(xx-padding)
    contour[1]=contour[1]-direction*(yy-padding)
    imageroi=image[yy-padding:yy+hh+2*padding, xx-padding:xx+ww+2*padding]
    return contour,imageroi

# def consistencyInsSeg(image,contour):
#     unfolded_images = unfold_wo_center(
#         image, kernel_size=3, dilation=2
#         )            
#     diff = abs(image[:, :, None] - unfolded_images)
#     unfolded_weights = torch.max(diff, dim=2)[0]
#     thred[thred<0.5]=0
#     loss1=unfolded_weights*thred

def predtoInstance(contours,temp_image):
    pred = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)  
    contoursList=sortBylen(contours)
    pred_instance=[None,]
    for kk in range(0, len(contoursList)):
        pts = np.asarray([contoursList[kk]], dtype=np.int32)
        # contour,imageroi=ConvertCoordinates(contour,1)
        # consistencyInsSeg(imageroi,contour)
        # if len(pts[0][0])<7:
        #     continue
        # pts1=grabCut1image(temp_image,pts)
        predi = np.zeros([temp_image.shape[0], temp_image.shape[1]], dtype=temp_image.dtype)
        predi = cv2.drawContours(predi,pts,-1,1,-1)
        pred = cv2.drawContours(pred,pts,-1,kk+1,-1)
        pred_instance.append(predi) 
    return pred_instance,pred
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
    
    solver = MyFrame(ResUNet34, dice_bce_loss, 3e-4)  
    solver.loadtar(Constants.weight_dir+'ResUNet34TargetA')#cenet\AL-Net_no_alloss_clusteredCell1 /home/chenxiaokai/daima/CE-Net-master/weights/U_Netttrain
    # testlist=['test-sample/','test-difficult/','test-normal/']#'test/',_kmeansAL_NET_clusteredCell
    # trainlist=['test-sample_json/','test-difficult_json/','test-normal_json/']#,
    # testlist=['sample/','difficult/','normal/']#'test/',_kmeansAL_NET_clusteredCell
    # trainlist=['sample_json/','difficult_json/','normal_json/']#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ,
    # testlist=['test_images/']#'test/',kmean
    # trainlist=['test_json/']#,
    testlist=['test-images/']#'test/',kmean
    trainlist=['test-labels/']#,
    # predlist=['/home/zhaojing/hover_net/output/epoch44/json//']#,'test_json/','/home/zhaojing/hover_net/output/epoch45/json/','/home/zhaojing/hover_net/output/epoch43/json/'
    # testlist=['test-sample/']
    # trainlist=['test-sample_json/']
    # testlist=['test-normal/']
    # trainlist=['test-normal_json/']
    # testlist=['test-difficult/']
    # trainlist=['test-difficult_json/']
    
    instaji=0
    instpq=0
    insthd=0
    instDC=0
    aji=0
    pq=0
    hd=0
    DC=0
    index=0
    nucleiNum=0
    for ii in range(0,len(testlist)):
        images_name = os.listdir(images_dir+testlist[ii])
        images_name = sorted(images_name)  # 对文件按照文件名进行排序
        
        for image_name in tqdm(images_name):  # 对于每一个文件进行处理image 是文件的名字
            imgpath=os.path.join(images_dir+testlist[ii])
            temp_image = cv2.imread(os.path.join(imgpath, image_name)) 
            imagesize = cv2.resize(temp_image, (size,size))    
            maskpath=images_dir+trainlist[ii]
            maskname=image_name.replace("jpg","json")
            # maskname=image_name.replace("png","json")
            # maskname=image_name.replace("png","xml")
           
            img = np.array(imagesize, np.float32).transpose(2, 0, 1) / 255.0* 3.2 - 1.6
            img = torch.Tensor(img)
            img=img.unsqueeze(0)
            solver.set_input(img)#
            # pred,ring = solver.test_image()# AL-Net
            pred=solver.test_image_noring()
            sr=pred.clone()
            # proced_pred,predimage=postprocess(sr,temp_image,ring)   # AL-Net 
            proced_pred,predimage=postprocess_noring(sr,temp_image)
            contour=extractContour(proced_pred)
            pred_instance,pred_mask=predtoInstance(contour,temp_image)
            # proced_pred,predimage=genpredImage(predlist[ii]+maskname,temp_image)# HoVer net
            true_instance,true_mask=genMask(temp_image,maskpath+maskname)
            # nucleiNum+=len(true_instance)
            # imagemask,mask=genXMLMask(temp_image,maskpath+maskname)
            # imagemask,mask=toMask(temp_image,maskpath+maskname)
            # imagemask = np.expand_dims(imagemask, axis=0)
            instaji+=instance_AJI(true_instance,pred_instance,true_mask)
            aji += get_fast_aji(true_mask,pred_mask)
            instpq+=instance_pq(true_instance,pred_instance,true_mask)
            pq+=get_fast_pq(true_mask,pred_mask)
            insthd+=instance_HD(true_instance,pred_instance,true_mask)
            instDC+=instance_Dice(true_instance,pred_instance,true_mask)
            true_mask=true_mask>0
            pred_mask=pred_mask>0
            DC+=Dice(true_mask,pred_mask)
            surface_distances = surfdist.compute_surface_distances(true_mask, pred_mask, spacing_mm=( 1.0,1.0))
            hd += surfdist.compute_robust_hausdorff(surface_distances, 95)
            # print('[test]  AJI: %.4f, instaji: %.4f,PQ: %.4f,instpq:%.4f,HD: %.4f,insthd: %.4f, DC: %.4f,  instdc:%.4f '%(aji,instaji,pq,instpq,hd,insthd,DC,instDC))

            index=index+1
    aji=aji/index
    pq=pq/index
    hd=hd/index
    DC=DC/index
    instaji=instaji/index
    instpq=instpq/index
    insthd=insthd/index
    instDC=instDC/index
    print('[test]  AJI: %.4f, instaji: %.4f,PQ: %.4f,instpq:%.4f,HD: %.4f,insthd: %.4f, DC: %.4f,  instdc:%.4f '%(aji,instaji,pq,instpq,hd,insthd,DC,instDC))
    # print('nucleiNum:%d'%(nucleiNum))


if __name__ == '__main__':
    
    # testNet('/home/zhaojing/clusteredCell/')
    testNet('/home/zhaojing/TargetA/')
    # testNet('/home/zhaojing/TargetB/')
    # testNet('/home/zhaojing/PatchSeg/')
    # testNet('/home/zhaojing/AL-Net/dataset/')
    