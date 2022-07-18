"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import random
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from PIL import Image
import albumentations as A
import cv2
import numpy as np
import os
import scipy.misc as misc

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,ring,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        ring = cv2.warpPerspective(ring, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask,ring

def randomShiftScaleRotate_k(image, mask,ring,kmeans,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        ring = cv2.warpPerspective(ring, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        kmeans = cv2.warpPerspective(kmeans, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask,ring,kmeans

def randomHorizontalFlip(image, mask,ring, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        ring = cv2.flip(ring, 1)
    return image, mask,ring

def randomVerticleFlip(image, mask, ring,u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        ring = cv2.flip(ring, 0)
    return image, mask,ring

def randomRotate90(image, mask,ring, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        ring=np.rot90(ring)
    return image, mask,ring

def randomHorizontalFlip_k(image, mask,ring, kmeans,u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        ring = cv2.flip(ring, 1)
        kmeans = cv2.flip(kmeans, 1)
    return image, mask,ring,kmeans

def randomVerticleFlip_k(image, mask, ring,kmeans,u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        ring = cv2.flip(ring, 0)
        kmeans = cv2.flip(kmeans, 0)
    return image, mask,ring,kmeans

def randomRotate90_k(image, mask,ring,kmeans, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        ring=np.rot90(ring)
        kmeans=np.rot90(kmeans)
    return image, mask,ring,kmeans

def default_loader(img_path, mask_path):

    img = cv2.imread(img_path)
    # print("img:{}".format(np.shape(img)))
    img = cv2.resize(img, (448, 448))

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = 255. - cv2.resize(mask, (448, 448))
    
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    
    mask = np.expand_dims(mask, axis=2)
    #
    # print(np.shape(img))
    # print(np.shape(mask))

    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    #mask = abs(mask-1)
    return img, mask
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
        A.RandomShadow (shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.5),
        A.RandomSnow (snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5),
        A.RandomSunFlare (flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), always_apply=False, p=0.5),
        A.RandomToneCurve (scale=0.1, always_apply=False, p=0.5),
        A.Solarize (threshold=128, always_apply=False, p=0.5),
        A.Blur (blur_limit=7, always_apply=False, p=0.5),
        A.Downscale (scale_min=0.25, scale_max=0.25, interpolation=0, always_apply=False, p=0.5),
        A.Equalize (mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.5),
        A.FancyPCA (alpha=0.1, always_apply=False, p=0.5),
        A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),
        A.GridDropout (ratio=0.5, unit_size_min=None, unit_size_max=None, holes_number_x=None, holes_number_y=None, shift_x=0, shift_y=0, random_offset=False, fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),
        A.RandomGridShuffle (grid=(3, 3), always_apply=False, p=0.5),
    ])
    transformed = transform(image=image)
    transformed_image = transformed["image"]
    return transformed_image

def default_DRIVE_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (448, 448))
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ringpath=mask_path.replace("masks","ring")
    # ringpath=mask_path.replace("labels","contours")
    imagering = np.array(Image.open(ringpath)) #cv2.imread(ringpath)
    imagering = cv2.resize(imagering, (448,448)) 
    mask = np.array(Image.open(mask_path))
    mask = cv2.resize(mask, (448, 448))
    # kmeanspath=mask_path.replace("masks","kmeans")
    # kmeansimg = cv2.imread(kmeanspath)
    # kmeansimg = cv2.resize(kmeansimg, (448, 448))
    img=colortransforms(img)
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask ,imagering= randomShiftScaleRotate(img, mask,imagering,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask ,imagering= randomHorizontalFlip(img, mask,imagering)
    img, mask ,imagering= randomVerticleFlip(img, mask,imagering)
    img, mask ,imagering= randomRotate90(img, mask,imagering)

    mask = np.expand_dims(mask, axis=2)
    imagering = np.expand_dims(imagering, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    imagering = np.array(imagering, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    imagering[imagering >= 0.5] = 1
    imagering[imagering <= 0.5] = 0
    # mask = abs(mask-1)
    img = torch.Tensor(img)
    mask = torch.Tensor(mask)
    imagering = torch.Tensor(imagering)
    mask = torch.cat([mask, imagering], dim=0)
    return img, mask
def kmeans_k(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 将图像重塑为像素和3个颜色值（RGB）的2D数组
    # print(image.shape) #(853, 1280, 3)
    pixel_values = image.reshape((-1, 3))
    # 转换为numpy的float32
    pixel_values = np.float32(pixel_values)
    # print(pixel_values.shape) #(1091840, 3)
    # 确定停止标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.1)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 转换回np.uint8
    centers = np.uint8(centers)

    # 展平标签阵列
    labels = labels.flatten()

    segmented_image = centers[labels.flatten()]
    #重塑回原始图像尺寸
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image


def default_kmeans_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (448, 448))
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    kmeanspath=mask_path.replace("masks","kmeans")
    kmeansimg = cv2.imread(kmeanspath)
    kmeansimg = cv2.resize(kmeansimg, (448, 448))

    ringpath=mask_path.replace("masks","ring")
    # ringpath=mask_path.replace("labels","contours")
    imagering = np.array(Image.open(ringpath)) #cv2.imread(ringpath)
    imagering = cv2.resize(imagering, (448,448)) 
    mask = np.array(Image.open(mask_path))
    mask = cv2.resize(mask, (448, 448))
    # img=kmeans_k(img)
    # cv2.imwrite("/home/zhaojing/AL-Net/1.jpg",img)
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask ,imagering,kmeansimg= randomShiftScaleRotate_k(img, mask,imagering,kmeansimg,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask ,imagering,kmeansimg= randomHorizontalFlip_k(img, mask,imagering,kmeansimg)
    img, mask ,imagering,kmeansimg= randomVerticleFlip_k(img, mask,imagering,kmeansimg)
    img, mask ,imagering,kmeansimg= randomRotate90_k(img, mask,imagering,kmeansimg)
    mask = np.expand_dims(mask, axis=2)
    imagering = np.expand_dims(imagering, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    kmeansimg = np.array(kmeansimg, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    imagering = np.array(imagering, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    imagering[imagering >= 0.5] = 1
    imagering[imagering <= 0.5] = 0
    # mask = abs(mask-1)
    img = torch.Tensor(img)
    kmeansimg = torch.Tensor(kmeansimg)
    img = torch.cat([img, kmeansimg], dim=0)
    mask = torch.Tensor(mask)
    imagering = torch.Tensor(imagering)
    mask = torch.cat([mask, imagering], dim=0)
    return img, mask
def default_3d_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    # img=img1[0:512, 0:500]
    img = cv2.resize(img, (448, 448))
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # kmeanspath=mask_path.replace("masks","kmeans")
    # kmeansimg = cv2.imread(img_path)
    kmeansimg=colortransforms(img)
    # kmeansimg=kmeansimg1[0:512, 12:512]
    # kmeansimg = cv2.resize(kmeansimg, (448, 448))
    # cv2.imwrite("/home/zhaojing/AL-Net/1.jpg",kmeansimg)
    # cv2.imwrite("/home/zhaojing/AL-Net/2.jpg",img)
    ringpath=mask_path.replace("masks","ring")
    # ringpath=mask_path.replace("labels","contours")
    imagering = np.array(Image.open(ringpath)) #cv2.imread(ringpath)
    imagering = cv2.resize(imagering, (448,448)) 
    mask = np.array(Image.open(mask_path))
    mask = cv2.resize(mask, (448, 448))
    # img=kmeans_k(img)
    # cv2.imwrite("/home/zhaojing/AL-Net/1.jpg",img)
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask ,imagering,kmeansimg= randomShiftScaleRotate_k(img, mask,imagering,kmeansimg,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask ,imagering,kmeansimg= randomHorizontalFlip_k(img, mask,imagering,kmeansimg)
    img, mask ,imagering,kmeansimg= randomVerticleFlip_k(img, mask,imagering,kmeansimg)
    img, mask ,imagering,kmeansimg= randomRotate90_k(img, mask,imagering,kmeansimg)
    mask = np.expand_dims(mask, axis=2)
    imagering = np.expand_dims(imagering, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    kmeansimg = np.array(kmeansimg, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    imagering = np.array(imagering, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    imagering[imagering >= 0.5] = 1
    imagering[imagering <= 0.5] = 0
    # mask = abs(mask-1)
    img = torch.Tensor(img)
    kmeansimg = torch.Tensor(kmeansimg)
    img = torch.cat([img, kmeansimg], dim=0)
    mask = torch.Tensor(mask)
    imagering = torch.Tensor(imagering)
    mask = torch.cat([mask, imagering], dim=0)
    return img, mask
def read_ORIGA_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'Set_A.txt')
    else:
        read_files = os.path.join(root_path, 'Set_B.txt')

    image_root = os.path.join(root_path, 'images')
    gt_root = os.path.join(root_path, 'masks')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.jpg')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.jpg')

        print(image_path, label_path)

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def default_simsiam_loader(img_path, mask_path):
    img = cv2.imread(img_path)
 
    img = cv2.resize(img, (448, 448))
 
    kmeansimg=colortransforms(img)
   
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    kmeansimg = np.array(kmeansimg, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
 
    # mask = abs(mask-1)
    img = torch.Tensor(img)
    kmeansimg = torch.Tensor(kmeansimg)
    img = torch.cat([img, kmeansimg], dim=0)

    return img, img

def read_Messidor_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train.txt')
    else:
        read_files = os.path.join(root_path, 'test.txt')

    image_root = os.path.join(root_path, 'save_image')
    gt_root = os.path.join(root_path, 'save_mask')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks

def read_RIM_ONE_datasets(root_path, mode='train'):
    images = []
    masks = []

    if mode == 'train':
        read_files = os.path.join(root_path, 'train_files.txt')
    else:
        read_files = os.path.join(root_path, 'test_files.txt')

    image_root = os.path.join(root_path, 'RIM-ONE-images')
    gt_root = os.path.join(root_path, 'RIM-ONE-exp1')

    for image_name in open(read_files):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.png')
        label_path = os.path.join(gt_root, image_name.split('.')[0] + '-exp1.png')

        images.append(image_path)
        masks.append(label_path)

    return images, masks


def read_DRIVE_datasets(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'training/images')
    gt_root = os.path.join(root_path, 'training/1st_manual')


    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name.split('.')[0] + '.tif')
        label_path = os.path.join(gt_root, image_name.split('_')[0] + '_manual1.gif')

        images.append(image_path)
        masks.append(label_path)

    print(images, masks)

    return images, masks
def read_clusteredCell_datasets(root_path, mode):
    images = []
    masks = []
    if mode=="test":
        list_name=['test-difficult','test-normal','test-sample']#,'test'
    elif mode =="train":
        list_name=['difficult','normal','sample']#,'train'

    for i in range(0,len(list_name)):
        image_root = os.path.join(root_path, list_name[i])#+'_kmeans'
        gt_root = os.path.join(root_path, list_name[i]+'_masks')
        for image_name in os.listdir(image_root):
            #if i%DATA_NUMBER==0 and mode=='train':
            image_path = os.path.join(image_root, image_name)
            label_path = os.path.join(gt_root, image_name)
            filename = image_name.split('.')
            GT_name =filename[0] + '.png'        
            label_path = os.path.join(gt_root, GT_name)        
            # data_ringuf_seg_loader(image_path, label_path)
            images.append(image_path)
            masks.append(label_path)
    print(mode)
    print(len(masks))
    return images, masks

def read_UNSUR_datasets(root_path, mode):
    images = []
    masks = []
    DATA_NUMBER=4
    index=0
    if mode=="test":
        list_name=['test-difficult','test-normal','test-sample']#,'test'
    elif mode =="train":
        list_name=['difficult','normal','sample']#,'train'

    for i in range(0,len(list_name)):
        image_root = os.path.join(root_path, list_name[i])#+'_kmeans'
        gt_root = os.path.join(root_path, list_name[i]+'_masks')
        for image_name in os.listdir(image_root):
            index=index+1
            if mode=="test":
                DATA_NUMBER=1
            if index%DATA_NUMBER==0:
                image_path = os.path.join(image_root, image_name)
                label_path = os.path.join(gt_root, image_name)
                filename = image_name.split('.')
                GT_name =filename[0] + '.png'        
                label_path = os.path.join(gt_root, GT_name)        
                # data_ringuf_seg_loader(image_path, label_path)
                images.append(image_path)
                masks.append(label_path)
    # if mode=='train':
    #     image_root = os.path.join('/home/zhaojing/AL-Net/dataset/jiarun1', mode+'_images')#mode+
    #     gt_root = os.path.join('/home/zhaojing/AL-Net/dataset/jiarun1', mode+'_masks') 
    #     for image_name in os.listdir(image_root):
    #         image_path = os.path.join(image_root, image_name)
    #         label_path = os.path.join(gt_root, image_name)
    #         filename = image_name.split('.')
    #         GT_name =filename[0] + '.png'        
    #         label_path = os.path.join(gt_root, GT_name)        
    #         images.append(image_path)
    #         masks.append(label_path)
    print(mode)
    print(len(masks))
    return images, masks

def read_data_datasets(root_path, mode='train'):
    images = []
    image_root = os.path.join(root_path, mode+'_images')#mode+

    
    for image_name in os.listdir(image_root):
        #if i%DATA_NUMBER==0 and mode=='train':
        # if image_name.rfind("_0.png")!=-1:
        image_path = os.path.join(image_root, image_name)
          
        images.append(image_path)
    
    print(mode)
    print(len(images))
    return images,images

def read_Cell_datasets(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, mode+'-images')#mode+
    # image_root = os.path.join(root_path, mode+'_kmeans')
    gt_root = os.path.join(root_path, mode+'-masks')
    # image_root = os.path.join(root_path, mode+'-images')#mode+
    # gt_root = os.path.join(root_path, mode+'-labels')
    
    for image_name in os.listdir(image_root):
        #if i%DATA_NUMBER==0 and mode=='train':
        # if image_name.rfind("_0.png")!=-1:
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)
        #image_name=image_name.replace('_fake_B','')
        filename = image_name.split('.')
        GT_name =filename[0] + '.png'        
        label_path = os.path.join(gt_root, GT_name)        
        images.append(image_path)
        masks.append(label_path)
    print(mode)
    print(len(masks))
    return images, masks


def read_datasets_vessel(root_path, mode='train'):
    images = []
    masks = []

    image_root = os.path.join(root_path, 'training/images')
    gt_root = os.path.join(root_path, 'training/mask')

    for image_name in os.listdir(image_root):
        image_path = os.path.join(image_root, image_name)
        label_path = os.path.join(gt_root, image_name)

        if cv2.imread(image_path) is not None:

            if os.path.exists(image_path) and os.path.exists(label_path):

                images.append(image_path)
                masks.append(label_path)

    print(images[:10], masks[:10])

    return images, masks


class ImageFolder(data.Dataset):

    def __init__(self,root_path, datasets='Messidor',  mode='train'):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ['RIM-ONE', 'simsiam', 'UNSUR', 'clusteredCell', 'Cell', 'Kmeans_Cell'], \
            "the dataset should be in 'Messidor', 'ORIGA', 'RIM-ONE', 'Vessel' "
        if self.dataset == 'RIM-ONE':
            self.images, self.labels = read_RIM_ONE_datasets(self.root, self.mode)
        elif self.dataset == 'simsiam':
            self.images, self.labels = read_data_datasets(self.root, self.mode)
        elif self.dataset == 'UNSUR':
            self.images, self.labels = read_UNSUR_datasets(self.root, self.mode)
        elif self.dataset == 'clusteredCell':
            self.images, self.labels = read_clusteredCell_datasets(self.root, self.mode)
        elif self.dataset == 'Cell':
            self.images, self.labels = read_Cell_datasets(self.root, self.mode)
        elif self.dataset == 'Kmeans_Cell':
            self.images, self.labels = read_Cell_datasets(self.root, self.mode)
        else:
            print('Default dataset is Messidor')
            self.images, self.labels = read_Messidor_datasets(self.root, self.mode)

    def __getitem__(self, index):
        # if self.mode=="train":
        # img, mask = default_simsiam_loader(self.images[index], self.labels[index])
        # else:
        img, mask = default_DRIVE_loader(self.images[index], self.labels[index])
        
       
        return img, mask

    def __len__(self):
        assert len(self.images) == len(self.labels), 'The number of images must be equal to labels'
        return len(self.images)