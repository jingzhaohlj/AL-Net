from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from re import M

import cv2
import os
from time import time

from numpy import random
import AJI
from evaluation import *
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable as V
from skimage import measure
from networks.cenet import *
from networks.alnet import *
from networks.attunet import *
from networks.u2net import *
from networks.nestednet import *
from networks.joinseg import *
from networks.nucleiSegnet import *
from framework import MyFrame
from loss import dice_bce_loss,dice_shape_loss
from data import ImageFolder
from tensorboardX import SummaryWriter
import Constants


# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def Net_Train():
    NAME = 'AL_NET_JR_' + Constants.ROOT.split('/')[-1]
    writer = SummaryWriter(Constants.log_dir+'AL_NET_en_JR/')
    solver = MyFrame(Model,dice_bce_loss, 2e-4)
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    dataset = ImageFolder(root_path=Constants.ROOT, datasets='simsiam',mode='train')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)
    

    loadepoch=0
    best_unet_score = 0.
    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    # solver.loadtar('/home/zhaojing/AL-Net/weights/AL_NET_simsiam_decoder_clusteredCell1b')
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    for epoch in range(1, total_epoch + 1):
        
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0
        ac = 0.	# Accuracy
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient  
        loadepoch=loadepoch+1
        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss= solver.optimize_simsiam()#_refalnetatloss,  
            train_epoch_loss += train_loss
            index = index + 1
          
        zloss=train_epoch_loss
        print(train_epoch_loss)
        solver.savetar(Constants.weight_dir + NAME + '1', loadepoch, train_epoch_loss) 
    
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '.th')
            # solver.update_lr(2.0, factor=True, mylog=mylog)
    solver.savetar(Constants.weight_dir + NAME + 'sz', total_epoch, zloss)          
    writer.close()
    print('Finish!')


if __name__ == '__main__':
    # print(torch.__version__())
    Net_Train()



