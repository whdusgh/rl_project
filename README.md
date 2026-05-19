# LLM-Guided Reinforcement Learning with Adaptive Trust Gate

## 프로젝트 개요

본 프로젝트는 LLM 기반 reasoning을 활용하는 강화학습(RL) 프레임워크를 구현하는 것을 목표로 한다.

핵심 아이디어는:

State
→ LLM reasoning 생성
→ Qwen reasoning encoder
→ Adaptive trust gate
→ PPO policy

구조를 통해, RL agent가 LLM의 조언을 무조건 신뢰하는 것이 아니라 현재 상황과 reasoning 품질에 따라 선택적으로 신뢰하도록 만드는 것이다.

---

# 현재 구현된 전체 파이프라인

MiniGrid 환경
↓
환경 상태 분석 및 텍스트 변환
↓
LLM 기반 Reasoning 생성
↓
Qwen Encoder를 통한 Reasoning Latent Encoding
↓
Adaptive Gate를 통한 Reasoning 신뢰도 조절 (Adaptive Gate: LLM 얼마나 믿을지)
↓
PPO 기반 Policy 학습

---

# 폴더 구조 및 역할

## agents/

### ppo_agent.py
Stable-Baselines3 기반 PPO 모델 생성.

### custom_ppo.py
PPO 내부 auxiliary optimization 실험용 파일.

---

## env/

### minigrid_env.py
MiniGrid 환경 생성.

### llm_wrapper.py
전체 시스템의 핵심 wrapper.

현재 수행 기능:
- state parsing (게임 화면 상태를 사람이 읽을 수 있는 문장으로)
- reasoning generation (LLM이 조언 생성)
- reasoning quality 계산 (좋은 reasoning인지 나쁜 reasoning인지 판단)
- multimodal observation 생성 (PPO가 사용할 최종 입력 데이터 생성)

Observation 구성:
- image
- reasoning embedding
- quality score

---

## llm/

### llm_generator.py
Qwen2-0.5B-Instruct 기반 reasoning 생성 모듈.

현재 특징:
- strategy pool 기반 reasoning 생성
- random bad reasoning injection 포함

예시:
- "Find the key first."
- "Move randomly."
- "Do nothing."

---

### qwen_encoder.py
Reasoning text를 latent embedding vector로 변환.

Reasoning text
→ Qwen encoder
→ 896차원 latent vector

---

### state_parser.py
MiniGrid observation을 text state로 변환.

예시:
- "A key is visible."
- "A door is visible."

---

### reasoning.py
현재 상황에서 이 조언이 맞는가? 판단 

예시:
- key visible + "Find the key first" → GOOD
- key visible + "Do nothing" → BAD

출력:
- 1.0 = good
- 0.0 = bad
- 0.5 = uncertain

---

## models/

### multimodal_extractor.py
PPO가 사용할 최종 feature 만드는 부분

현재:
- image feature
- reasoning latent feature

를 결합하여 PPO 입력으로 사용.

또한:
- adaptive gate network (LLM 조언을 얼마나 믿을지)
- auxiliary trust supervision (gate가 잘 배우도록 도와주는 추가 학습)

구현 포함.

현재 gate 구조:

gate_value =
0.8 * quality
+
0.2 * predicted_gate

---

### custom_cnn.py
이미지 feature extraction용 CNN backbone.

---

# 테스트 파일 설명

### test_parser.py
State parsing 테스트.

### test_llm.py
Reasoning generation 테스트.

### test_qwen.py
Qwen latent encoding 테스트.

### test_wrapper.py
전체 wrapper pipeline 테스트.

### test_extractor.py
Multimodal extractor 동작 테스트.

---

# 현재까지 구현 완료된 내용

## 완료된 항목

- PPO 강화학습 baseline 구현
- LLM 기반 reasoning(전략/조언) 생성 구현
- Qwen을 이용한 reasoning embedding 생성
- 이미지 정보 + reasoning 정보 결합 구현
- LLM 조언을 얼마나 믿을지 결정하는 adaptive gate 구현
- 현재 상황(state)에 따라 reasoning 품질 평가 구현
- gate 학습을 위한 추가 loss(lossB) 적용
- PPO와 reasoning 시스템 연결 완료
- 일부러 잘못된 reasoning(BAD reasoning) 생성 기능 추가
- 좋은 reasoning / 나쁜 reasoning에 따라 gate 값이 달라지는 behavior 확인

현재 관찰된 behavior:
- GOOD reasoning → 높은 gate
- BAD reasoning → 낮은 gate

즉:
reasoning 품질에 따라 trust level이 달라지는 behavior가 나타나기 시작함.

---

# 앞으로 해야 할 주요 작업

## 1. Baseline 비교 실험

비교 필요:

### A. PPO only
image → PPO

### B. PPO + fixed gate
gate = quality

### C. PPO + adaptive gate
learned trust estimation

비교 지표:
- reward
- success rate
- learning speed

---

## 2. 시각화

필요한 그래프:
- reward curve
- gate trajectory
- reasoning trust behavior

---

## 3. PPO 내부 loss 통합

현재는 auxiliary loss가 PPO optimization과 부분적으로 분리된 상태.

향후:
- cleaner PPO loss integration
- joint optimization

필요.

---

## 4. 더 큰 LLM 실험


---

# 핵심 연구 아이디어

본 프로젝트는 단순히:
"LLM을 RL에 사용"

하는 것이 아니라,

"RL agent가 상황과 reasoning 품질에 따라 LLM guidance를 선택적으로 신뢰"

하도록 만드는 것이 핵심 목표이다.

즉:
- LLM 조언을 상황에 따라 다르게 신뢰
- reasoning이 믿을만한지 평가
- 현재 상황에 따라 조언 활용 방식이 달라짐

를 핵심 연구 포인트로 둔다.

---

# 실행 방법

## 학습

```bash
python train.py
```

## 개별 테스트

```bash
python test_llm.py
python test_parser.py
python test_qwen.py
python test_wrapper.py
```

