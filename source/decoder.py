
import torch
from torch import nn
from torch.nn import functional as f
from attention import selfattention

class VAEattentionblock(nn.Module):

  def __init__(self,channels : int):
    super().__init__()
    self.groupnorm = nn.GroupNorm(32,channels)
    self.attention = selfattention(1,channels)

  def __call__(self,x: torch.Tensor) -> torch.Tensor:

    residue = x

    batchsize,c,h,w = x.shape

    x =  x.view(batchsize,c,h*w)

    # bs,features,h*w -> bs,h*w,features h*w 는 sequence, sequence마다 각각의 features를 가지고 있음
    # -> selfattention 으로 sequence간의 관련성
    x = x.transpose(-1,-2)

    x = self.attention(x)

    x = x.transpose(-1,-2)

    x = x.view((batchsize,c,h,w))

    x += residue

    return x

class VAEresidualblock(nn.Module):

  def __init__(self,inchannels,outchannels):
    super().__init__()
    self.groupnorm1 = nn.GroupNorm(32,inchannels)
    self.conv1 = nn.Conv2d(inchannels,outchannels,kernel_size=3,padding=1)
    self.groupnorm2 = nn.GroupNorm(32,outchannels)
    self.conv2 = nn.Conv2d(outchannels,outchannels,kernel_size=3,padding=1)

    if inchannels == outchannels:
      self.residual_layer = nn.Identity()
    else: # inchannel outchannel 맞춰주기 위함
      self.residual_layer = nn.Conv2d(inchannels,outchannels,kernel_size=1)

  def forward(self,x:torch.Tensor) -> torch.Tensor:

    # x = batchsize,inchannels,height,width
    residue = x

    x = self.groupnorm1(x)

    x = f.silu(x)

    x = self.conv1(x)

    x = self.groupnorm2(x)

    x = f.silu(x)

    x = self.conv2(x)

    return x + self.residual_layer(residue)
    #inchannel outchannel 같으면 nn.identity <- 아무 변화 없음
    #다른경우 conv2d layer

class VAEdecoder(nn.Sequential):
  def __init__(self):
    super().__init__(
      nn.Conv2d(4,4,kernel_size=1),

      nn.Conv2d(4,512,kernel_size=3,padding=1),

      VAEresidualblock(512,512),

      VAEattentionblock(512),

      VAEresidualblock(512,512),
      VAEresidualblock(512,512),
      VAEresidualblock(512,512),
      VAEresidualblock(512,512),

      #batchsize,512,height/8,width/8 -> batchsize,512,height/4,width/4
      nn.Upsample(scale_factor=2),
      # 가로 세로 크기 2배로 늘림
      # 기본으로 nearest 방식을 사용, 가장 빠르지만 이미지 품질이 저하될수있음
      nn.Conv2d(512,512,kernel_size=3,padding=1),

      VAEresidualblock(512,512),
      VAEresidualblock(512,512),
      VAEresidualblock(512,512),
      #batchsize,512,h/4,w/4 -> batchsize,512,h/2,w/2
      nn.Upsample(scale_factor=2),

      nn.Conv2d(512,512,kernel_size=3,padding=1),

      VAEresidualblock(512,256),
      VAEresidualblock(256,256),
      VAEresidualblock(256,256),

      #batchsize,512,h/2,w/2 -> batchsize,512,h,w
      nn.Upsample(scale_factor=2),

      nn.Conv2d(256,256,kernel_size=3,padding=1),

      VAEresidualblock(256,128),
      VAEresidualblock(128,128),
      VAEresidualblock(128,128),

      nn.Groupnorm(32,128),

      f.silu(),

      #batchsize,128,height,width -> batchsize,3,height,width
      nn.Conv2d(128,3,kernel_size=3,padding=1)
    )
  
  def __call__(self,x : torch.Tensor) -> torch.Tensor:
    # batchsize, 4, h/8, w/8
    x/= 0.18215
    for module in self:
      x = module(x)

    # batchsize, 3, h, w
    return x
     
