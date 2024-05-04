참고 https://seastar105.tistory.com/176
https://www.youtube.com/watch?v=5ZSwYogAxYg
https://www.youtube.com/watch?v=7NNxK3CqaDk
stable diffusion 3 - rectified flow
https://arxiv.org/pdf/2403.03206

rectified flow 
https://openreview.net/pdf?id=PqvMRDCJT9t

visual autoregressive modeling - gpt based 
https://arxiv.org/abs/2404.02905

![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/d3d38a72-7df4-4535-a973-2bde07f7cecb)


![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/a664b75d-e8a4-4e2b-9a79-8785ecbe2bc0)

Diffusion model은 데이터를 noise와 반복적으로 혼합하여 최종적으로 pure noise 상태로 만든 뒤, 이 과정을 역방향으로 재현하여 원본 데이터를 복원하는 원리로 동작

이미지에 noise를 넣다 뺏다하는 과정을 통해 이미지들의 distribution을 학습함
우리가 쓰는 이미지 생성은 여기 과정에 CLIP을 통해 텍스트 데이터를 추가해 학습시킨 것

"flow matching"은 모델이 데이터 분포의 전/후방향 과정(forward/reverse process)에서 발생하는 확률 흐름(probability flow)을 정확히 따라가도록 학습시키는 것을 의미합니다.
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/f4f4daff-7135-42bc-847e-fca591ecb7c1)

여기서 사용된 플로우 매칭 <-- Continuous Normalizing Flows 를 시뮬레이션 없이 효율적으로 훈련하는 방법
rectified flow 논문 내용
플로우 매칭은 Gaussian probability paths (노이즈와 데이터 샘플사이에서의)과 양립할 수 있다
claude 피셜
Probability path는 이러한 점진적인 노이즈 추가 및 제거 과정에서 거치는 확률 분포들의 궤적(trajectory)을 말합니다. 구체적으로 다음과 같은데

초기 데이터 분포 q(x0)에서 시작합니다.
노이즈를 점진적으로 추가하면서 q(x1|x0), q(x2|x1), ..., q(xT|xT-1) 와 같은 조건부 분포를 거칩니다. - markov chain
마지막에는 q(xT) 와 같이 노이즈만 있는 가우시안 분포에 이릅니다.
이제 역방향으로 p(xT-1|xT), p(xT-2|xT-1), ..., p(x0|x1)의 조건부 분포를 따라가며 노이즈를 제거합니다
 p()와 q()는 얘네가 "probability path" 같음, 구성하는? 확률 분포
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/4c58c48f-1b8b-4b6a-a90e-5e5525acbe0d)Normalizing Flow는 위 그림과 같이 대부분의 생성 모델은 우리가 잘 알고 샘플링하기 쉬운 분포 z(노이즈)에서 생성하기 원하는 분포 x(input:입력이미지)
로의 변환을 학습하게 된다. Normalizing Flow는 데이터 분포인 x에서 z로의 역변환(reverse pass)이 가능한 flow를 뉴럴 네트워크 가 학습하는 것이 목적
뉴럴 네트워크 f는 역변환(reverse pass)이 가능하기 때문에 생성 시에는 z에서 뽑은 샘플 z0에 f^-1(역함수)를 적용하게 되면 output 생성이 가능해짐 

CNF는 Neural Ordinary Differential Equations[^4]에서 제시됐는데 뉴럴넷이 변환 자체인 flow를 학습하는 것이 아니라, flow의 벡터장(vector field)를 학습하는 것

![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/0fee2084-5de3-4db1-80a3-f6b05baac986)
단점 <- 적분 때문에 매우느림
![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/ce43acc4-39a5-4f69-97a5-b65206e8fd2a)


노이즈 분포 p1에서 샘플 x1과 데이터 분포 p0에서 샘플 x0 사이의 매핑을 정의하는 생성 모델 <- diffusion model 이 매핑은 일반적인 미분 방정식(ODE)의 형태로 표현됩니다.

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
