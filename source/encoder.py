import torch
from torch import nn
from torch.nn import functional as f
from decoder import VAEattentionblock, VAEresidualblock
class VAEencoder(nn.Sequential):
  def __init__(self):
    super.__init__(

        # bs,c,h,w => bs,128,h,w
        nn.Conv2d(3,128,kernel_size=3,padding=1),
        VAEresidualblock(128,128),
        VAEresidualblock(128,128),

        # bs,128,h,w => bs,128,h/2,w/2 채널수가 변하지않아서 h,w가 1/2배됨
        nn.Conv2d(128,128,kernel_size=3,stride=2),
        VAEresidualblock(128,256),
        VAEresidualblock(256,256),

        #bs,256,h/2,w/2 > bs,256,h/4,w/4
        nn.Conv2d(256,256,kernel_size=3,stride=2),  #nn.conv2d padding 기본값은 0
        VAEresidualblock(256,512),
        VAEresidualblock(512,512),

        VAEattentionblock(512),
        VAEresidualblock(512,512),

        nn.GroupNorm(32,512),
        nn.SiLU(),

        nn.Conv2d(512,8,kernel_size=3,padding=1),
        nn.Conv2d(8,8,kernel_size=1), #커널사이즈 1 shape 변화 없음


        #bs,256,h/4,w/4 > bs,256,h/8,w/8
        nn.Conv2d(512,512,kernel_size=3,stride=2),
        VAEresidualblock(256,512),
        VAEresidualblock(512,512),
        VAEresidualblock(512,512),
    )
  def forward(self, x : torch.Tensor, noise : torch.Tensor) -> torch.Tensor:
    # x : bs, c, h,w
    # noise : bs, out_channels, h/8,w/8
    for module in self: # 상속받은 layers
      if getattr(module,'stride',None) == (2,2):  #stride 옆으로 2칸, 밑으로 2칸씩 이동
        # padding (좌,우,상,하)
        f.pad(x,(0,1,0,1))
      x= module(x)

  #다변량 정규분포의 mean variance, latent space
    # (bs,8,h/8,w/8) -> (bs,4,h/8,w/8)
    mean , log_var = torch.chunk(x,2,dim=1)

    # -30,20 사이의 값으로 clamp
    log_var = torch.clamp(log_var,-30,20)

    var = log_var.exp()
    #std
    std = var.sqrt()

    # Z = N(0,1) -> N(mean,variance)
    mean + std * noise

    #scale output
    x *= 0.18125 #경험론적 방법에 의해 얻어진 scale 상수

    return x
