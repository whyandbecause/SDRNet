import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lib.SDRNet import Network
from torch import nn
from Src.utils.Dataloader import test_dataset1
import cv2
import imageio
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from torch import cosine_similarity

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='/dataset2/gjw/SDRNet/checkpoints/SDRNet/Net_epoch_best.pth')
opt = parser.parse_args()

model = Network().cuda()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()
S=0
#for _data_name in ['CAMO', 'COD10K', 'CHAMELEON','NC4K']:
for _data_name in [ 'COD10K']:
    data_path = '/dataset2/gjw/TestDataset/{}'.format(_data_name)
    save_path = './result/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name+'_S')
    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root',image_root,gt_root)
    test_loader = test_dataset1(image_root=image_root,
                               gt_root=gt_root,
                               edge_root=gt_root,
                               testsize=opt.testsize)
    print('****',test_loader.size)
    with torch.no_grad():
      for i in range(test_loader.size):
          image, gt, name = test_loader.load_data()
          gt = np.asarray(gt, np.float32)
          image = image.cuda()
          cam, _1, _2, _3,_,_,_,_ = model(image)
          cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=False)
          res = cam.sigmoid().data.cpu().numpy().squeeze()#
          res = (res - res.min()) / (res.max() - res.min() + 1e-8)
          imageio.imsave(save_path+name, img_as_ubyte(res))
      
          
    