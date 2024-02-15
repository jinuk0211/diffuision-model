import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMsampler

width = 512
height = 512
latents_width = width // 8
latents_height = height // 8
#strength - 처음 이미지에 얼마만큼의 attention을 부여할것인가
# cfg_scale - prompt 에 얼마만큼의 value를 부여할것인가
# idle_device - 여러 GPU 또는 TPU가 사용 가능한 경우 프레임워크는 작업을 효율적으로 분산하기 위해 자원 분배 최적화  device 의 균등한 활용
# device - device to create tensor
def generate(prompt:str,
             uncond_prompt:str,
             inputimage=None,
             strength=0.8,
             do_cfg=True,cfg_scale=7.5,
             sampler_name='ddpm',
             numinferencesteps=50,
             models={},
             seed=None,
             device=None,idle_device=None,
             tokenizer=None):
  with torch.no_grad(): #inferencing 과정
    if not (0 <strength <=1): #ex) strength가 1.5, -1.0 등일때
      raise ValueError('strength는 0과 1사이의 값이여야 합니다.')

    if idle_device:
      to_idle: lambda x: x.to(idle_device) #idle_device에 자원할당

    else:
      to_idle : lambda x:x #그냥 놔둠
  generator = torch.Generator(device=device)
  if seed is None:
    generate.seed()
  else:
    generator.manual_seed(seed)

  clip = models['clip']
  clip.to(device)

                        # prompt              #negative prompt 
            # w=cgf_scale                       # 이미지와 상관없는 prompt
    #### output = w *(conditioned_output - unconditioned_output) +unconditioned_output
  
  if do_cfg:
    #일단 tokenizer로 prompt 변환 
    cond_tokens = tokenizer.batch_encode_plus([prompt],padding='max_length',max_length=77).input_ids
    # batchsize,seqlen
    cond_tokens = torch.tensor(cond_tokens,dtype=torch.long,device=device)
    #CLIP 통과 batchsize,seqlen -> batchsize,seqlen,embd
    cond_context = clip(cond_tokens)

    uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt],padding='max_length',max_length=77).input_ids
    uncond_tokens = torch.tensor(uncond_tokens,dtype=torch.long,device=device)
    uncond_context = clip(uncond_tokens)    

    context = torch.cat([cond_context,uncond_context])
    #shape (2,77,768)
 

  else:
    tokens = tokenizer.batch_encode_plus([prompt],padding='max_length',max_length=77).input_ids
    tokens = torch.tensor(tokens,dtype=torch.long,device=device)
    context = clip(tokens)
    #shape(1,77,768)

    #prompt txt 데이터 처리 과정
  to_idle(clip)

  if sampler_name == 'ddpm':
    sampler = DDPMsampler(generator)
    sampler.set_inference_step(numinferencesteps)
  else:
    raise ValueError('알 수 없는 샘플러입니다')

  latents_shape = (1,4,latents_height,latents_width)

  if inputimage:
    encoder = models['encoder']
    encoder.to(device)

    inputimagetensor = inputimage.resize((width,height))
    inputimagetensor = np.array(inputimagetensor)
    #h,w,c
    inputimagetensor = torch.tensor(inputimagetensor,dtype=torch.float32)

    #UNET에 맞는 tensor shape 형태로 변환
    inputimagetensor= rescale(inputimagetensor,(0,255),(-1,1))
    #batch 차우너 추가
    inputimagetensor = inputimagetensor.unsqueeze(0)    
    # bs,h,w,c -> bs,c,h,w - encoder tensor 형식
    inputimagetensor = inputimagetensor.permute(0,3,1,2)
    # Sampling noise                          #특정 seed값을 갖고 있는 generator
    encoder_noise = torch.randn(latents_shape,generator=generator,device=device) #1,4,h,w
    
    #VAE encoder 실행
    latents = encoder(inputimagetensor,encoder_noise)

    sampler.set_strength(strength=strength)
    # strength 가 커질수록 noise 증가, the less output resembles to input
    latents = sampler.add_noise(latents,sampler.timesteps[0]) #encoder통과한후 noise가 추가됨   

    #encoder 역할 종료
    to_idle(encoder)
  else:
    # text-to-img를 랜덤한 정규분포 noise N(0,I)로 시작 
    latents = torch.randn(latents_shape,generator=generator,device=device)
    
  diffusion = models['diffusion']
  diffusion.to(device)


# timestep은 생성된 이미지나 텍스트의 점진적 정교화 단계를 나타냄.
# timestep이 감소하면서 노이즈는 점진적으로 제거되고 최종 이미지나 텍스트의 기본 구조와 세부 사항이 나타남
  timesteps = tqdm(sampler.timesteps)
  for i,timestep in enumerate(timesteps):
    # (1,320)
    time_embedding = get_time_embedding(timestep).to(device)

    # (batchsize, 4, h/8-latents height, w/8 - latents width)
    model_input = latents

    if do_cfg: # conditioned uncond 토큰화된 프롬프터 벡터

      # batchsize,4,latents_height,latents_width -> batchsize*2,...
      model_input = model_input.repeat(2,1,1,1) 
      # 2를 곱하는 이유 cond,uncond 두개의 concat된 prompt tensor와 차원을 맞춤

    model_output = diffusion(model_input,context,time_embedding)

    if do_cfg:
      condoutput, uncondoutput = model_output.chunk(2)

      # 위의 공식
      model_output = cfg_scale*(condoutput-uncondoutput) + uncondoutput

# schedular에 의해 denoise됨 UNET안에서

    latents = sampler.step(timestep,latents,model_output)

  to_idle(diffusion)

  decoder = models['decoder']
  decoder.to(device)

  images = decoder(latents)

  # 채널값이 0~255에서 -1~1로 rescale된거를 복구
  images = rescale(images,(-1,1),(0,255))
  # c,h,w -> h,w,c
  images = images.permute(0,2,3,1)
  images = images.to('cpu',torch.uint8).numpy()
  return images[0] # image의 c,h,w 

def rescale(x,oldrange,newrange,clamp=False):
  oldmin, oldmax = oldrange #(0,255) 
  newmin, newmax = newrange #(-1,1)
  x -= oldmin #0을 중심으로 맞춤
  x *= (newmax-newmin) / (oldmax - oldmin)
  x += newmin #새 범위내에 배치
  if clamp:
    x= x.clamp(newmin,newmax) #모든 값을 newmin newmax사이 값으로 변환
  return x

def get_time_embedding(timestep):
  #160,
  freqs = torch.pow(10000,-torch.arange(start=0,end=100,dtype= torch.float32)/160)
  (1,160)
  x = torch.tensor([timestep],dtype=torch.float32)[:,None]*freqs[None] #None은 새로운 차원추가
  # timestep이 1차원이면 [:,None] 을 함으로써 2차원이됨

  return torch.cat([torch.cos(x),torch.sin(x)],dim=-1)
