import os
import shutil

Image_size = (448, 448)#
ROOT="/home/zhaojing/AL-Net/dataset/jiarun1"
# ROOT = '/home/zhaojing/dataset/clusteredCell'
# ROOT = '/repository01/nucleus_seg_data/Uniform1'
# ROOT = '/repository01/nucleus_seg_data/BNS'
# ROOT = '/repository01/nucleus_seg_data/MoNuSeg'
BATCHSIZE_PER_CARD = 2
TOTAL_EPOCH = 300
INITAL_EPOCH_LOSS = 10000
NUM_EARLY_STOP = 20
NUM_UPDATE_LR = 10
log_dir='./logs/'
weight_dir='./weights/'
BINARY_CLASS = 1