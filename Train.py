
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from torch import optim
from torchvision.utils import make_grid
import utils.python.metrics as Measure
from utils.utils import clip_gradient
import torch
from Src.utils.Dataloader import get_loader, test_dataset
import torch.nn.functional as F
import os

from torch import nn
import logging
import torch.backends.cudnn as cudnn
from lib.SDRNet import Network
import pytorch_ssim
import cv2
from torchsummary import summary

def structure_loss(pred, mask):

    a = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) + 1e-6
    b = torch.abs(F.avg_pool2d(mask, kernel_size=51, stride=1, padding=25) - mask) + 1e-6
    c = torch.abs(F.avg_pool2d(mask, kernel_size=61, stride=1, padding=30) - mask) + 1e-6
    d = torch.abs(F.avg_pool2d(mask, kernel_size=27, stride=1, padding=13) - mask) + 1e-6
    e = torch.abs(F.avg_pool2d(mask, kernel_size=21, stride=1, padding=10) - mask) + 1e-6
    alph = 1.75
    
    fall = a**(1.0/(1-alph)) + b**(1.0/(1-alph)) + c**(1.0/(1-alph)) + d**(1.0/(1-alph)) + e**(1.0/(1-alph))
    a1 = ((a**(1.0/(1-alph))/fall)**alph)*a
    b1 = ((b**(1.0/(1-alph))/fall)**alph)*b
    c1 = ((c**(1.0/(1-alph))/fall)**alph)*c
    d1 = ((d**(1.0/(1-alph))/fall)**alph)*d
    e1 = ((e**(1.0/(1-alph))/fall)**alph)*e

    weight = 1 + 5* (a1+b1+c1+d1+e1)
    
    dwbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    dwbce = (weight * dwbce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weight).sum(dim=(2, 3))
    union = ((pred + mask) * weight).sum(dim=(2, 3))
    dwiou = 1 - (inter + 1) / (union - inter + 1)
    
    return (dwbce + dwiou).mean()  

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()
    
def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function

    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):
            
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()
            s_1, s_2, s_3, d_1, d_2, d_3  = model(images)
            
            loss1s = structure_loss(s_1, gts)
            loss2s = structure_loss(s_2, gts)
            loss3s = structure_loss(s_3, gts)
            loss_s =  loss1s + 0.5*loss2s + 0.25*loss3s
            
            loss1d = dice_loss(d_1, edges)
            loss2d = dice_loss(d_2, edges)
            loss3d = dice_loss(d_3, edges)
            loss_d =  loss1d + 0.5*loss2d + 0.25*loss3d

            loss = loss_s + loss_d
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data



            if i % 40 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_d: {:.4f}  loss_s: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_d.data, loss_s.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_d: {:.4f} loss_s: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_d.data, loss_s.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'loss_d': loss_d.data, 'loss_s': loss_s.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)


        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, _, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            #print(np.min(gt))
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            
            res, _0, _1, _2, _3,_4= model(image)
            
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))
        
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
       

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='/dataset2/gjw/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/dataset2/gjw/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='train use gpu')
    parser.add_argument('--save_epoch', type=int, default=5,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_path', type=str, default='./checkpoints/SDRNet/',
                        help='the path to save model and log')

    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')
    cudnn.benchmark = False

    # build the model
    model = Network().cuda()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Image/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = test_dataset(image_root=opt.val_root + 'Image/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)


    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0
    best_mae = 0
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=5e-6)
    #multi_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50, 70], gamma=0.1)
    print(">>> start train...")
    for epoch in range(1, opt.epoch):
        # schedule
        save_path = opt.save_path
        os.makedirs(save_path, exist_ok=True)

        if (epoch + 1) % opt.save_epoch == 0:
            torch.save(model.state_dict(), save_path + 'LNet_%d.pth' % (epoch + 1))
        cosine_schedule.step()
        #multi_scheduler.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # train
        #from torch import autograd
        #with autograd.detect_anomaly():
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)




