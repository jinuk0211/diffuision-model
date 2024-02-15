import torch
import numpy as np
class DDPMsampler: #Denoising Diffusion Probabilistic Model
  def __init__(self, generator= torch.Generator,
               numtrainingsteps=1000,
               betastart:float =0.00085,
               betaend:float = 0.0120):    #parameter는 순방향과정
    # beta start   #역방향과정 
  #  DDPM 모델 학습 시작 시 노이즈 강도를 조절하는 하이퍼파라미터
  #  높은 값은 더 많은 노이즈를 의미하며, 모델 학습 초기에는 이미지 정보 손실을 방지하기 위해 높은 값을 사용하는 것이 일반적
  #  DDPM 모델 학습 종료 시 노이즈 강도를 조절하는 하이퍼파라미터입니다.
  # 낮은 값은 더 깨끗한 이미지 생성을 의미하며, 모델 학습 후반에는 원본 이미지에 가까운 결과를 얻기 위해 낮은 값을 사용하는 것이 일반적
    self.betas  = torch.linspace(betastart**0.5,betaend**0.5,numtrainingsteps)
    # Alpha는 0과 1 사이의 값을 가지며, 낮은 값은 더 많은 정보 손실을, 높은 값은 더 적은 정보 손실을 의미함.
    self.alphas = 1 - self.betas
    # DDPM 모델의 역방향 프로세스에서 사용됨.
    # 각 노이즈 단계에서 얼마나 많은 정보를 유지할지를 결정하는 역할을 합니다.
    self.alphaprod1 = torch.cumprod(self.alphas,0) # x = torch.tensor([1, 2, 3, 4])- cumprod - # tensor([ 1,  2,  6, 24])
    self.one = torch.tensor(1.0)                      #noise 수학식에 사용됨

    self.generator = generator
    self.numtrainingsteps = numtrainingsteps
    self.timesteps= torch.from_numpy(np.arange(0,numtrainingsteps)[::-1].copy())

  def set_inference_timesteps(self,numinferencesteps=50):
    self.numinferencesteps = numinferencesteps
    step_ratio = self.numtrainingsteps // self.numinferencesteps
    timesteps = (np.arange(0,numinferencesteps)*step_ratio).round()[::-1].copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps)

  def add_noise(self,sample,timesteps):   #timestep마다 noise를 추가하는 과정
    alphaprod1 = self.alphaprod1.to(device=sample.device,dtype=sample.dtype)
    timesteps = timesteps.to(sample.device)
    alphaprod2 = alphaprod1[timesteps]**0.5 #  noise gaussian 분포 mean 부분에 활용됨
    alphaprod2 = alphaprod2.flatten()
    while len(alphaprod1.shape) < len(sample.shape):
      alphaprod2 = alphaprod2.unsqueeze(-1)

    alphaprod3 = (1-alphaprod1[timesteps]) ** 0.5 # noise gaussian 분포 variance부분에 활용됨
    alphaprod3 = alphaprod3.flatten()
    while len(alphaprod3.shape) < len(sample.shape):
      alphaprod3 = alphaprod3.unsqueeze(-1)


    noise = torch.randn(sample.shape,generator=self.generator, device = sample.device,dtype=sample.dtype)
    noisysample = (alphaprod2 * sample) + (alphaprod3 * noise)
    return noisysample


  def get_previous_step(self,timestep: int) -> int:
    prevt = timestep - (self.numtrainingsteps // self.numinferencesteps)
    return prevt


  def get_variance(self,timestep):
    prevt = self.get_previous_step(timestep)
    alphaprod_t = self.alphaprod[timestep]
    alphaprod_prevt = self.alphaprod[prevt] if prevt >=0 else self.one 
    beta_t = 1 - alphaprod_t / alphaprod_prevt #current beta t formula에 사용되는

    var = (1- alphaprod_prevt ) / (1 - alphaprod_t) * beta_t
    var = torch.clamp(var,min=1e-20)

    return var

  def step(self,timestep,latents,model_output): # unet의 결과물인 predicted noisysample을 denoise해서 반복함
    t = timestep
    prevt = self.get_previous_step(t)

    alphaprod_t = self.alphaprod[timestep]
    alphaprod_prevt = self.alphaprod[prevt] if prevt >=0 else self.one 
    betaprod_t = 1- alphaprod_t
    betaprod_prevt = 1- alphaprod_prevt
    alpha_t = alphaprod_t / alphaprod_prevt #진짜 alpha t formula에 사용되는
    beta_t =1 - alpha_t #curretnt
    
    #unet에 의해 생성된 sample
    pred_sample = (latents - betaprod_t ** 0.5 * model_output) / alphaprod_t ** 0.5
    
    
    #coefficient 구하기
    pred_sample_coeff = (alphaprod_prevt ** 0.5 * beta_t) / beta_t
    current_sample_coeff = alpha_t ** 0.5 * betaprod_prevt / betaprod_t

    #predicted presample 의 평균을 구해보자
    pred_presample = pred_sample_coeff * pred_sample + current_sample_coeff * latents
    
    var = 0
    if t>0:
      device = model_output.device
      noise = torch.randn(model_output,generator =self.generator,device=device,dtype=model_output.dtype)
      var = (self.get_variance(t)**0.5) * noise #std * noise

      #N(0,1) ->  N(mu,sigma^2)
      pred_presample = pred_presample + var
    return pred_presample

    #image2image
  def set_strength(self,strength=1):
    startstep = self.numinferencesteps -int(self.numinferencesteps * strength )
    self.timesteps = self.timesteps[startstep:]
    self.startstep = startstep







