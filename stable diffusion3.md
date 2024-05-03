![image](https://github.com/jinuk0211/diffuision-model/assets/150532431/a664b75d-e8a4-4e2b-9a79-8785ecbe2bc0)

Diffusion model은 데이터를 noise와 반복적으로 혼합하여 최종적으로 pure noise 상태로 만든 뒤, 이 과정을 역방향으로 재현하여 원본 데이터를 복원하는 원리로 동작

이미지에 noise를 넣다 뺏다하는 과정을 통해 이미지들의 distribution을 학습함
우리가 쓰는 이미지 생성은 여기 과정에 CLIP을 통해 텍스트 데이터를 추가해 학습시킨 것

플로우매칭은 diffusion 과정에서 noise가 추가될 때마다 데이터 분포의 변화량을 최소화하도록 모델을 학습시키는 기법
이를 통해 모델이 원본 데이터 분포를 보다 정확히 학습할 수 있음
CFM(Continuous Flow Matching)은 이런 플로우매칭 기법 중 하나로,
데이터와 noise 사이의 전이 확률을 연속적인 함수로 모델링하여 분포 변화를 줄임. 이는 diffusion 과정의 연속성을 높여 모델 성능을 향상시킴

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
