참고 https://seastar105.tistory.com/176
https://www.youtube.com/watch?v=5ZSwYogAxYg
https://www.youtube.com/watch?v=7NNxK3CqaDk
stable diffusion 3 - rectified flow
https://arxiv.org/pdf/2403.03206

CFM - conditional flow matching
https://openreview.net/pdf?id=PqvMRDCJT9t
 CNFs are capable of modeling arbitrary probability path and are in particular known to encompass the probability paths modeled by diffusion processes
 
1.디퓨전 denoising score matching 를 제외하고는 효율적인 훈련 알고리즘의 부재
2.기존의  maximum likelihood training의 경우 비싼 ODE simulations를
3. existing simulation-free methods는  intractable integrals 또는 biased gradients를 포함했다
이 논문에서는 이를 위해 FLOW MATCHNG 을 제안
이 flow matching을 통해 CNF (continous normalizing flow)를 학습
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/ec024b63-98b2-44b8-a90d-28bf88e48eb3)

![Uploading image.png…]()

이 때 flow matching의 학습하는 주체는 벡터장으로 
이 벡터장이 probability path를 생성

claude 피셜
Probability path는 이러한 점진적인 노이즈 추가 및 제거 과정에서 거치는 확률 분포들의 궤적(trajectory)을 말합니다. 구체적으로 다음과 같은데

초기 데이터 분포 q(x0)에서 시작합니다.
노이즈를 점진적으로 추가하면서 q(x1|x0), q(x2|x1), ..., q(xT|xT-1) 와 같은 조건부 분포를 거칩니다. - markov chain
마지막에는 q(xT) 와 같이 노이즈만 있는 가우시안 분포에 이릅니다.
이제 역방향으로 p(xT-1|xT), p(xT-2|xT-1), ..., p(x0|x1)의 조건부 분포를 따라가며 노이즈를 제거합니다

학습된 벡터장은 desired probability path를 생성할 수 있게 됨 
<- 노이즈에서 우리가 원하는 이미지를 생성할 수 있게된다는 뜻

이 벡터장은 flow matching 논문에 의하면
per-example (i.e., conditional) formulations로 구하는게 가능함

이 벡터장 학습시키는 공식은 score matching 노이즈 없애는 것에 영감을 받은 
per-example training objective로 이게 == CFM (conditional flow matching)

장점 

위의 3의 문제를 해결 
1. equivalent gradients 제공
2. explicit knowledge of the intractable target vector field를 필요로 하지 않음

위의 1의 문제 해결
diffusion paths에 관해서도 score matching과 비교해 매우 우수

더 빠른 훈련시간, 짧아진 생성시간, 좋은 퍼포먼스
 이러한 probability path 의(family)에는 매우 흥미로운 case 가 포함됨 :
 그것이 최적 수송(Optimal Transport, OT) 변위 보간법(displacement interpolant)(McCann, 1997)에 상응(correspond)하는 벡터장
 이러한 conditonal OT path가 diffusion path보다 더 단순한 형태임을 발견.  OT 경로(path)는 직선 궤적(trajectory)을 형성하는 반면, 디퓨전 path(경로)는 곡선 궤적(path)을 만듦

https://arxiv.org/abs/2404.02905

![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/d3d38a72-7df4-4535-a973-2bde07f7cecb)


![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/a664b75d-e8a4-4e2b-9a79-8785ecbe2bc0)

Diffusion model은 데이터를 noise와 반복적으로 혼합하여 최종적으로 pure noise 상태로 만든 뒤, 이 과정을 역방향으로 재현하여 원본 데이터를 복원하는 원리로 동작

이미지에 noise를 넣다 뺏다하는 과정을 통해 이미지들의 distribution을 학습함
우리가 쓰는 이미지 생성은 여기 과정에 CLIP을 통해 텍스트 데이터를 추가해 학습시킨 것

"flow matching"은 모델이 데이터 분포의 전/후방향 과정(forward/reverse process)에서 발생하는 확률 흐름(probability flow)을 정확히 따라가도록 학습시키는 것을 의미합니다.
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/f4f4daff-7135-42bc-847e-fca591ecb7c1)

여기서 사용된 플로우 매칭 <-- Continuous Normalizing Flows 를 시뮬레이션 없이 효율적으로 훈련하는 방법

![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/4c58c48f-1b8b-4b6a-a90e-5e5525acbe0d)Normalizing Flow는 위 그림과 같이 대부분의 생성 모델은 우리가 잘 알고 샘플링하기 쉬운 분포 z(노이즈)에서 생성하기 원하는 분포 x(input:입력이미지)
로의 변환을 학습하게 된다. Normalizing Flow는 데이터 분포인 x에서 z로의 역변환(reverse pass)이 가능한 flow를 뉴럴 네트워크 가 학습하는 것이 목적
뉴럴 네트워크 f는 역변환(reverse pass)이 가능하기 때문에 생성 시에는 z에서 뽑은 샘플 z0에 f^-1(역함수)를 적용하게 되면 output 생성이 가능해짐 

CNF는 Neural Ordinary Differential Equations[^4]에서 제시됐는데 뉴럴넷이 변환 자체인 flow를 학습하는 것이 아니라, flow의 벡터장(vector field)를 학습하는 것

 p()와 q()는 얘네가 "probability path" 을 구성하는? 확률분호 그리고 벡터장이 이 desired probablity path를 생성함

그리고 이 벡터장을 모르기 때문에 flow matching의 수식을 통해 구해냄
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/0fee2084-5de3-4db1-80a3-f6b05baac986)
단점 <- 적분 때문에 매우느림
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/ce43acc4-39a5-4f69-97a5-b65206e8fd2a)
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/246b93fc-a5c5-4bf9-b0c7-6bbc6e618e0a)


노이즈 분포 p1에서 샘플 x1과 데이터 분포 p0에서 샘플 x0 사이의 매핑을 정의하는 생성 모델 <- diffusion model 이 매핑은 일반적인 미분 방정식(ODE)의 형태로 표현
but

![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/4d4c0ce8-e840-4d78-a7ed-3ce6aad27f8d)

dyt = vΘ(yt, t) dt , (1)
여기서 velocity v는 신경망의 가중치 Θ에 의해 매개변수화(paramatized)
Chen 등(2018)의 선행 연구에서는 미분 가능한 ODE solver를 통해 방정식 (1)을 직접 풀 것을 제안
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/a26eb9f8-0a2e-48c9-984b-aab5463a260a) 
걍 미분으로 파라미터를 구한다고 생각하면 됨?
SDE 설명
시간 t에 따라 변하는 각 state들을 표현하는 process로, 이 시간에 따라 변하는 결과는 deterministic하지 않고 randomness가짐
즉, 시간이 흐름에 따라 process가 항상 같은 것이 아니라 randomness에 의해 조금씩 달라짐. SDE의 기본적인 형태는 밑과 같다.
wiener process라고도 불리는데 diffusion에서는 noise의 식을 의미
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/186749b2-d9cb-48a1-8c70-62b22f8029b4)



그러나 이 과정은 vΘ(yt, t)를 매개변수화하는 대규모 네트워크 아키텍처에서 cost 높음
보다 효율적인 대안은 p0와 p1 사이의 확률 경로(prob path)를 생성하는 벡터장(vector field) ut를 직접 regress 
이 ut를 생성하기 위해, forward pass라는 것을 정의한다.(reverse path를 위해 forward pass를 정의 -> 벡터장 ut)
이를 위해 우리는 p0과 p1 = N(0, 1) 사이의 확률 경로 pt에 해당하는 전방 과정을 정의합니다. 
pt (p0 ~ p1) 식 : (2)
