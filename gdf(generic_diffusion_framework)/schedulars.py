import torch 
import numpy as np 

class baseschedule(): #뼈대 모든 schedular
  def __init__(self,*args,forcelimits=True,discretesteps=None,shift=1,**kwargs):
    self.setup(*args,**kwargs) 
    self.limits = None
    self.discretesteps = discretesteps #learninig rate 단위별 step
    self.shift = shift 
    if forcelimits:
      self.reset_limits() #내장함수
  def reset_limits(self,shift=1,disable=False):
    try:                                          #args setup 에 속함
      self.limits = None if disable else self(torch.tensor([1.0, 0.0]), shift=shift).tolist() # min, max
