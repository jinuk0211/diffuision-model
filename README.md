# diffuision-model
webui 한국어버젼 
CLIP prompt - korean
deep danbooru 개량 예정

either
website - domain 구입 - aws web hosting - api

website2 - gradio제공 domain사용 -backend- ,user authentication,postsql, or complex routing functionalities.
ex) instagram 5 million user 시절 아키텍쳐
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/c06f63f0-2e88-4d5e-9c48-fdc078d23468)


![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/4ba23b38-c887-40c9-a023-228332ca89f3)



diffusion model architecture 
- 스테이블 디퓨전
- 그냥 이미지를 cnn layer unet 으로 upsample downsample 하던 것을
- latent space에 noise를 추가하고 denoise함으로써 학습시키는 것으로 만듦
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/142b7b85-a86f-42dd-b82c-f9a726ec37f6)


SDXL turbo

stable diffusion model에서 unet을 2개로 늘림 base 역할의 unet, refiner역할의 unet 사용
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/906fc39e-093c-42ec-a540-496981109bd1)

cascade 
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/5ab437ab-f857-48ee-9604-f5f9c2ab326e)


-------------extension----------------

realesrgan - enhanced super resolution GAN
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/67880487-2e9a-43f8-a884-b9ab95bee6fa)


gfpgan - generative facial prior GAN
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/a25723ea-7681-4ab0-85f7-4fe9766ab4c0)


codeformer - 얼굴 복원
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/e0a32057-6856-46b3-a59b-4cfdcfc57d18)


IP adapter
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/0683be60-044e-4503-aabb-f09a8df7f413)


hypernetwork - parameter업데이트 방식
residual block 과 유사하지만 residual connection은 특정 feature의 벡터가 다음 트랜스포머 블록에 추가되지만 hypernetwork는  가공 전의 transformer 블록으로 다시 돌아가 추가된다는 차이가 있다
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/261d4ad3-f400-4c6c-b2fb-3d36b1c045c4)



backbone controlnet
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/cc9b8061-5000-4e8f-9430-3be5a48dc608)


hypernetwork 최적화 도움 도구 feed forward layer*
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/d1073c6a-1a2d-4083-b290-0966e303ebe8)


textual inversion stable diffusion model 파인튜닝
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/96b30974-1e8c-4f94-a5d1-c75bcfd45ab3)


LoRA train - 적은수의 파라미터 효율성
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/cfd93a91-ae5a-4ab7-a0d0-05e7efd3fc3f)


xlmr 
text classification 다국적 encoder
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/d96869ab-1e14-4b13-81d6-e8a5d445dec8)



clip - Contrastive Image-Language Pretraining
embedding architecture 
- 텍스트, 이미지 데이터 cross attention
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/019aa3f3-4a06-425a-86cc-57a081198a33)

-variatial autoencoder | VAE - latent space 가 핵심, 이미지의 고차원 joint distribution
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/65a05778-d728-4f2e-9fac-963a67c3e157)

 
VQGAN 
Vector Quantized GAN
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/0ff5d58a-a867-4f6b-90a2-2f65af0a46da)


번외
OPENAI SORA
diffusion transformer (DiT)
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/c5fb2609-2a7b-492a-8781-34e965f6dc82)
-> 비디오의 압축된 latent space
-> 시공간 patch를 사용해 트랜스포머 token으로 사용
-> sd와 같이 노이즈를 masking한뒤 노이즈 없는 이미지로의 예측으로 train
이 과정에서 transformer가 사용?
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/884d6eb2-cc75-4b6e-a473-085ceb59b2d6)
