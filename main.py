
# coding: utf-8

# ## Reproduce the result of *SmoothGrad: removing noise by adding noise*
# 
# Link to the paper: https://arxiv.org/pdf/1706.03825.pdf

# In[8]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import grad,Variable

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image


# load names of imagenet 1000 classes

class Saliency(object):
    def __init__(self,model):
        self.model = model
    def getMask(self,img,pred_label=None):
        pass
    def getSmoothMask(self,img,stddev=0.15,nsamples=20):
        stddev *= img.max()-img.min()
        total_grad = torch.zeros(img.shape)
        noise = torch.zeros(img.shape)
        for i in range(nsamples):
            img_plus_noise = img + noise.zero_().normal_(0,stddev) 
            grad = self.getMask(img_plus_noise)
            total_grad += grad * grad
        total_grad /= nsamples
        return total_grad

class VallinaSaliency(Saliency):
    def getMask(self,img,pred_label=None):
        self.model.eval()
        self.model.zero_grad()
        x = Variable(preprocess(img),
                     requires_grad=True)
        p = self.model(x)
        if pred_label is None:
            _,pred_label = torch.max(p,1)
            pred_label = pred_label.data
        pk = p[0][pred_label]
        pk.backward()
        
        mask3d = x.grad.data[0].cpu()
        return mask3d


    
def visualize_image_gray_scale(img3d,percentile=99):
    if str(type(img3d)).find('torch')!= -1:
        img3d = img3d.numpy()
        img3d = img3d.transpose([1,2,0])
    img2d = np.sum(np.abs(img3d),2)
    
    vmax  = np.percentile(img2d,percentile)
    vmin  = img2d.min()
    img2d = np.clip((img2d-vmin)/(vmax-vmin),0,1)
    return img2d


def test_image(model,img,smooth=False,nsamples=20,stddev=0.15):
    saliency = VallinaSaliency(model)
    if not smooth: 
        mask3d = saliency.getMask(img)
    else:
        mask3d = saliency.getSmoothMask(img,nsamples=nsamples,stddev=stddev)
    mask2d = visualize_image_gray_scale(mask3d,percentile=99)

    prob_pred = infer(model,img)
    label = prob_pred.argmax()

    plt.figure(figsize=(10,3));
    plt.subplot(1,3,1); imshow(img,classes[label])
    plt.subplot(1,3,2); plt.imshow(mask2d); plt.axis('off')
    plt.subplot(1,3,3); plt.plot(prob_pred,'o');

