import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torchvision
import torch
from matplotlib import pyplot as plt
from lib.SDRNet import Network
import numpy as np
def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens,size))
    if titles == False:
        titles="0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.02, hspace=0)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig("./CAM/"+fname, bbox_inches='tight')
    #plt.show()
def tensor2img(tensor,heatmap=False,shape=(224,224)):
    np_arr=tensor.detach().numpy()#[0]
    
    if np_arr.max()>1 or np_arr.min()<0:
        np_arr=np_arr-np_arr.min()
        np_arr=np_arr/np_arr.max()
    #np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0]==1:
        np_arr=np.concatenate([np_arr,np_arr,np_arr],axis=0)
    np_arr=np_arr.transpose((1,2,0))
    return np_arr
#input_tensors=torch.cat([input_tensor, input_tensor.flip(dims=(3,))],axis=0)
 
model_weight_path = "/dataset2/gjw/SDRNet/checkpoints/SDRNet/Net_epoch_best.pth"
model = Network().cuda()
#model = Hitnet()
model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

target_layers = [model.out_s4]

path="/dataset2/gjw/TestDataset/COD10K/Image/"
imgs = os.listdir(path)
for img_na in imgs:
  
  bin_data=torchvision.io.read_file(path+img_na)
  img=torchvision.io.decode_image(bin_data)/255
  img=img.unsqueeze(0)
  input_tensors=torchvision.transforms.functional.resize(img,[224, 224])
  #print(type(input_tensors))
  #cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)
  with GradCAM(model=model, target_layers=target_layers) as cam:
      #targets = [ClassifierOutputTarget(386),ClassifierOutputTarget(386)] 
      # aug_smooth=True, eigen_smooth=True 
      #print(input_tensors.shape)
      grayscale_cams = cam(input_tensor=input_tensors, targets=None)
      print(input_tensors.shape)
      for grayscale_cam,tensor in zip(grayscale_cams,input_tensors):
          
          rgb_img=tensor2img(tensor)
          visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
          myimshows([rgb_img, grayscale_cam, visualization],["image","cam","image + cam"], fname=img_na)
          #myimshows([visualization],["image + cam"], fname=img_na)
  

