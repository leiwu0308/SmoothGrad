import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import json,pprint


mean,std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MEAN = torch.Tensor(mean).unsqueeze(1).unsqueeze(2)
STD  = torch.Tensor(std).unsqueeze(1).unsqueeze(2)


# Processing

def preprocess(img0,USE_GPU=True):
    img = (img0-MEAN)/STD
    if img.dim()==3:
        img = img.unsqueeze(0)
    if USE_GPU:
        img  = img.cuda()
    return img

def deprocess(img):
    img = img.cpu()
    img = img*STD + MEAN
    img = img.squeeze()
    return img

class ClipImage:
    def __init__(self,img,epsilon,use_gpu=True):
        if epsilon > 1:
            epsilon = epsilon / 255.0
        self.imgL = img - epsilon
        self.imgU = img + epsilon
        self.mean = MEAN
        self.std  = STD
        if use_gpu:
            self.mean = MEAN.cuda()
            self.std  = STD.cuda()
            self.imgL = self.imgL.cuda()
            self.imgU = self.imgU.cuda()


    def __call__(self,img):
        img = img * self.std + self.mean
        img = torch.max(img,self.imgL)
        img = torch.min(img,self.imgU)
        img = img.clamp(0,1)
        img = (img-self.mean)/self.std
        return img



# Imagenet class name

json_data=open('imagenet_class_index.json').read()
data = json.loads(json_data)
idx2class = [data[str(k)][1] for k in range(1000)]



# ## Helper function
def PIL2tensor(img):
    img0 = img.resize((224,224),Image.BILINEAR)
    img0 = np.asarray(img0,dtype=np.float32)/255
    img0 = torch.from_numpy(img0.transpose([2,0,1])).float()
    return img0

class Infer:
    def __init__(self,use_gpu):
        self.use_gpu=use_gpu
    def __call__(self,model,img):
        model.eval()
        x = preprocess(img,self.use_gpu)
        img = Variable(x)
        p = model(img)
        p = p.data.squeeze().cpu().numpy()
        return p

def imshow_th(img,title=None):
    img_ = img.numpy().transpose([1,2,0])
    plt.imshow(img_)
    if title is not None:
        plt.title(title)
    plt.axis('off')



# Ensemble of networks
class Ensemble(nn.Module):
    def __init__(self,model_list):
        super(Ensemble,self).__init__()
        self.model_list = model_list
        self.numModel = len(model_list)

    def forward(self,x):
        pL = [model(x) for model in self.model_list]
        p  = sum(pL)/self.numModel
        return p



if __name__ == '__main__':
    print(MEAN)
    img = torch.rand(3,224,224)
    img = preprocess(img,False)
    print(img.shape)
