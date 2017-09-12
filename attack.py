import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import preprocess, deprocess,ClipImage


def target(p,idx=None):
    if idx is None or len(idx)==0:
        p0,idx = torch.max(p,1)
        return  torch.mean(p0)
    
    E = 0
    for i in idx:
        if p.dim()==1:
            p0 = p[i]
        else:
            p0 = p[:,i]
        E += p0
    E  = E/ len(idx)
    return - E

def negative_entropy(p):
    o = p*torch.log(p)
    E = torch.sum(o)
    return E


class Attack(object):
    def __init__(self,use_gpu=True):
        self.use_gpu = use_gpu

    def __call__(self,model,criterion,img,
                    epsilon=8,dt=2,nstep=1):
        model.eval()
        clip_img = ClipImage(img,epsilon/255.0,self.use_gpu)

        img_    = preprocess(img,self.use_gpu)
        x       = Variable(img_, requires_grad=True)
        dt      = dt / 255.0 * (img_.max()-img_.min())

        for _ in range(nstep):
            model.zero_grad()
            p = model(x)
            E = criterion(p)
            E.backward()

            x.data -= dt * x.grad.data.sign()
            x.data = clip_img(x.data)

        img_adv = deprocess(x.data)
        return img_adv

class GradAttack(object):
    def __init__(self,use_gpu=True):
        self.use_gpu = use_gpu

    def __call__(self,model,ct,img,epsilon=8,dt=2,nstep=1):
        model.eval()



if __name__ == '__main__':
    attack = Attack(True)
    print(attack)

