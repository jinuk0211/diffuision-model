import torch
from torch import nn
from torch.nn import functional as f
from attention import selfattention,crossattention


  
#더욱 정교한 이미지 제어: 시간 정보를 통해 모델은 이미지 생성 과정의 각 단계에서 어떤 특징을 강조할지 결정
#더욱 안정적인 이미지 생성: 시간 정보를 통해 모델은 이미지 생성 과정을 더욱 안정적으로 수행
#더욱 빠른 이미지 생성: 시간 정보를 통해 모델은 이미지 생성 과정을 더욱 빠르게 수행

class TimeEmbedding(nn.Module):
  def __init__(self,embd):
    super().__init__()
    self.linear1 = nn.Linear(embd,4*embd)
    self.linear2 = nn.Linear(4*embd,4*embd)

  def forward(self,x:torch.Tensor):
    # x :(1,320)
    x = self.linear1(x)
    x = f.silu(x)
    x = self.linear2(x)
    # 1,1280
    return x

class switchsequential(nn.Sequential):
                    #x = latent
  def forward(self, x,context,time) -> torch.Tensor:
    for layer in self:
      # switchsequential이라는 클래스를 정의하고 이를 nn.Sequential에서 상속받
      # switchsequential instance self는 nn.Sequential
      if isinstance(layer,UNETattentionblock): #image , text crossatteniton
        x= layer(x,context)
      elif isinstance(layer,UNETresidualblock): #image, time token embedding
        x = layer(x,time)
      else:
        x = layer(x)
    return x



class UNETresidualblock(nn.Module):
                                                #time embedding + image
  def __init__(self,inchannels,outchannels,n_time = 1280):
    super().__init()
    self.groupnorm1 = nn.GroupNorm(32,inchannels)
    self.conv1 = nn.Conv2d(inchannels,outchannels,kernel_size=3,padding=1)
    self.lineartime = nn.Linear(n_time,outchannels)  #x.shape @ tensor(n_time,outchannels)
    
    self.groupnorm2 = nn.GroupNorm(32,outchannels)
    self.conv2 = nn.Conv2d(outchannels,outchannels,kernel_size=3,padding=1)

    if inchannels == outchannels:
      self.residual_layer = nn.Identity()
    else: 
      self.residual_layer = nn.Conv2d(inchannels,outchannels,kernel_size=1,padding=0) 

  def forward(self,feature,time):
    # feature - bs,inchannels,height,width
    # time 1,1280
    residue = feature

    feature = self.groupnorm1(feature)

    feature = f.silu(feature)

    feature = self.conv1(feature)

    time = f.silu(time)

    time = self.lineartime(time)

    merged = feature + time.unsqueeze(-1).unsqueeze(-1) #time embedding이 batchsize inchannels 없음으로 만들어줌

    merged = self.groupnorm2(merged)

    merged = f.silu(merged)

    merged = self.conv2(merged)

    return merged + self.residual_layer(residue)

class UNETattentionblock(nn.Module):
  
  def __init__(self,numheads,n_dim,d_context=768):
    super().__init__()
    channels = numheads*n_dim

    self.groupnorm = nn.GroupNorm(32,channels,eps=1e-6)
    self.conv_input = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
    
    self.layernorm1 = nn.Layernorm(channels)
    self.attention1 = selfattention(numheads,channels,in_bias=True)
    self.layernorm2 = nn.Layernorm(channels) 
    self.attention2 = crossattention(numheads,channels,d_context,in_bias=False)
    self.layernorm3 = nn.Layernomr(channels)
    self.linear_geglu1 = nn.Linear(channels,4*channels*2)
    self.linear_geglu2 = nn.Linear(4* channels,channels)
    # 선형 GEGLU는 Gaussian Error Linear Unit (GELU)활성화 함수와 선형 변환을 결합한 신경망 레이어

    self.conv_output = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
  def forward(self,x):
    #x는 잠재벡터 batchsize, feature, height , width
    #context는 text prompt 벡터 batchsize,seqlen,dim
    longresidue = x
    x = self.groupnorm(x)

    x = self.conv_input(x)

    n, c, h, w = x.shape

    #normalization- selfattention- 잔차연결
     
    x = x.reshape(n,c,h*w)
    x = x.transpose(-1,-2)

    shortresidue = x

    x = self.layernorm(x)
    self.attention1(x)
    x += shortresidue

    #normalization- crossattention- 잔차연결

    shortresidue = x

    x = self.layernorm2(x)
    self.attention2(x,context)
    x += shortresidue

    #normalization- FF with GeGLU - 잔차연결

    shortresidue = x

    x = self.layernorm3(x)
    x,gate = self.linear_geglu1(x).chunk(2,dim=-1)
    x = x* f.gelu(gate)

    x = self.linear_geglu2(x)
    
    x += shortresidue

    #원래 shape로
    x = x.transpose(-1,-2)
     
    x = x.view((n,c,h,w))

    return self.conv_output(x) + longresidue





class Upsample(nn.Module): #nn.Upsampling의 
  def __init__(self,channels):
    super().__init__()
    self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

  def __call__(self,x): #h,w 2배
    x = f.interpolate(x,scale_factor = 2,mode='nearest')
    return self.conv(x)

class UNEToutputlayer(nn.Module):
  def __init__(self,inchannels,outchannels):
    super().__init__()
    self.groupnorm = nn.GroupNorm(32,inchannels)
    self.conv = nn.Conv2d(inchannels,outchannels,kernel_size=3,padding=1)

  def __call__(self,x):
    # x = batchsize,320,h/8,w/8 unet의 마지막 xshape
    x = self.groupnorm(x)
    
    x = f.silu(x)

    x = self.conv(x)  

    #bs, 4, h/8, w/8
    return x



class UNET(nn.Module): #기본적으로 encoder - bottleneck - decoder구조 그 중간의 skip connection

  def __init__(self):
    super().__init__()
    self.encoder = nn.Module([
        #batchsize 4 h/8 w/8

        switchsequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),

        switchsequential(UNETresidualblock(320,320),UNETattentionblock(8,40)),

        switchsequential(UNETresidualblock(320,320),UNETattentionblock(8,40)),

        #batchsize 320 h/16 w/16
        switchsequential(nn.Conv2d(320,320,kernel_size=2,padding=1)),
        
        switchsequential(UNETresidualblock(320,640),UNETattentionblock(8,80)),

        switchsequential(UNETresidualblock(320,640),UNETattentionblock(8,80)),

        #batchsize 640 h/32 w/32
               
        switchsequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),

        switchsequential(UNETresidualblock(640,1280),UNETattentionblock(8,160)),

        switchsequential(UNETresidualblock(1280,1280),UNETattentionblock(8,160)),

        #batchsize 1280 h/64 w/64

        switchsequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),

        switchsequential(UNETresidualblock(1280,1280)),
        #batchsize 1280 h/64 w/64

        switchsequential(UNETresidualblock(1280,1280)),
    ])

    self.bottleneck = switchsequential(
        UNETresidualblock(1280,1280),
        UNETattentionblock(8,160),
        UNETresidualblock(1280,1280),
    )

    self.decoder = nn.Modulelist([
     
    #bs,2560,h/64,w/64 2560인 이유 skip connection에 의해서 residue가 더해짐 
    #bs,1260,h/64,w/64
      switchsequential(UNETresidualblock(2560,1280)),
      switchsequential(UNETresidualblock(2560,1280)),
      switchsequential(UNETresidualblock(2560,1280),Upsample(1280)),
      switchsequential(UNETresidualblock(2560,1280),UNETattentionblock(8,160)),
      switchsequential(UNETresidualblock(2560,1280),UNETattentionblock(8,160)),
      switchsequential(UNETresidualblock(1920,1280),UNETattentionblock(8,160),Upasample(1280)),
      switchsequential(UNETresidualblock(1920,640),UNETattentionblock(8,80)),
      switchsequential(UNETresidualblock(1280,640),UNETattentionblock(8,80)),
      switchsequential(UNETresidualblock(960,640),UNETattentionblock(8,80),Upsample(640)),
      switchsequential(UNETresidualblock(960,320),UNETattentionblock(8,40)),
      switchsequential(UNETresidualblock(640,320),UNETattentionblock(8,40)), #80
      switchsequential(UNETresidualblock(640,320),UNETattentionblock(8,40)),
    
    ])


class diffusion(nn.Module):

  def __init__(self):
    self.timeembedding = TimeEmbedding(320)
    self.unet = UNET()
    self.final = UNEToutputlayer(320,4)

  def forward(self,latent : torch.Tensor, context: torch.Tensor, time: torch.Tensor):
    # latent - batchsize, 4 ,h/8, w/8
    # - encoder 결과물
    # - 이미지의 multivariational gaussian 분포
    # - 잠재벡터

    # context - batchsize,seqlen,dim - CLIP 결과물
    # time - (1,320)
    time = self.timeembedding

    #batchsize, 4, h/8, w/8 -> batchsize, 320, h/8, w/8
    output = self.unet(latent,context,time)

    output = self.final(output)

    return output













