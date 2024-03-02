import torch
class simplesampler(): #모든 sampler의 base형태
  def __init__(self,gdf):
    self.gdf
    self.currentstep= -1
  def __call__(self,*args,**kwargs):
    self.currentstep += 1
    return self.step(*args,**kwargs)
  def init_x(self,shape):
    return torch.randn(*shape) #unpacking
  def step(self,x,x0,epsilon,logSNR,logSNRprev):
    raise NotImplementedError('override가 필요한 function입니다')

class DDIMsampler(simplesampler) #signal to noise
  def step(self,x,x0,epsilon,logSNR,logSNRprev,eta=1):
    a,b = self.gdf.inputscaler(logSNR) 
    if len(a.shape) == 1:
      pass
class DDPMsampler(DDIMsampler):
  def step:
    pass

class LCMsampler(simplesampler): 
  def step(self,x,x0,epsilon,logSNR,logSNRprev):
    pass
