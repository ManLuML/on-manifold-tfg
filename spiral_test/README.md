# Spiral Test — JiT + Train-Free Guidance (TFG) Toy Experiment

2D toy 데이터셋을 이용하여 **prediction target (x, e, v)** 별 diffusion model의 Train-Free Guidance 성능을 비교하는 실험입니다.

## Overview

이 실험은 다음 질문에 답하기 위해 설계되었습니다:

> **어떤 prediction target이 Train-Free Guidance에 가장 유리한가?**

### 실험 설계

1. 2D toy 데이터 (e.g., double spiral)를 생성
2. 2D → D차원으로 random orthogonal projection (D = 2, 8, 32, 128, 512)
3. 각 D에 대해 **classifier** + **3종 diffusion model** (x-pred, e-pred, v-pred) 학습
4. DSP (Denoising Score-based Prior) guidance로 특정 클래스 방향 샘플 생성
5. 2D로 재투영 후 시각화하여 guidance 효과 비교

### 파이프라인

```
generate_data.py → train.py → inference.py
   (데이터 생성)     (모델 학습)   (샘플링 + 시각화)
```

---

## 1. 데이터 생성 (`generate_data.py`)

### 지원 데이터셋

| 이름 | 설명 | 클래스 수 | 기본 noise |
|------|------|-----------|-----------|
| `doublespiral` | 두 개의 interleaved spiral | 2 | 0.5 |
| `concentric_rings` | 동심원 (반지름비 1:2) | 2 | 0.1 |
| `circular_gaussians` | 원형 배치된 8개 Gaussian (4 클래스) | 4 | 0.3 |
| `grid_gaussians` | 3×3 격자 Gaussian | 9 | 0.2 |
| `crossed_lines` | X자 교차 직선 | 2 | 0.1 |
| `half_arcs` | 상하 반원 arc 2개 | 2 | 0.1 |

### 사용법

```bash
# 기본 (doublespiral)
uv run python spiral_test/generate_data.py --name doublespiral

# 다른 데이터셋
uv run python spiral_test/generate_data.py --name circular_gaussians

# 커스텀 파라미터
uv run python spiral_test/generate_data.py \
    --name doublespiral \
    --total_points 20000 \
    --noise 0.3 \
    --data_range 3.0 \
    --seed 42
```

### 출력 파일

```
spiral_test/data/<dataset_name>/
├── data.npz            # points_2d (N,2), labels (N,)
├── class_info.json     # 클래스 수, 색상 등 메타데이터
├── config.json         # 생성 시 사용한 파라미터
└── visualization.png   # 데이터 산점도
```

---

## 2. 모델 학습 (`train.py`)

각 D 값에 대해 다음을 학습합니다:
- **Classifier** — MLP 기반 (binary: sigmoid, multi-class: softmax)
- **Diffusion Models** — x-pred, e-pred, v-pred 각 1개씩 (총 3개)

### Flow Matching 정의

```
Forward process:  z_t = t · x + (1 - t) · ε
Prediction target:
  x-pred → network가 clean data x를 예측
  e-pred → network가 noise ε를 예측
  v-pred → network가 velocity v = x - ε를 예측
```

### 사용법

```bash
# 기본 설정 (D=2,8,32,128,512, 500 epochs)
uv run python spiral_test/train.py --data doublespiral

# D 값 선택
uv run python spiral_test/train.py --data doublespiral --d_values 2 8 16

# 에폭 수 조정
uv run python spiral_test/train.py \
    --data doublespiral \
    --diffusion_epochs 1000 \
    --classifier_epochs 200

# CPU 사용
uv run python spiral_test/train.py --data doublespiral --device cpu
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--data` | `doublespiral` | 데이터셋 이름 |
| `--d_values` | `2 8 32 128 512` | 학습할 차원 목록 |
| `--diffusion_epochs` | `500` | Diffusion 학습 에폭 |
| `--classifier_epochs` | `100` | Classifier 학습 에폭 |
| `--seed` | `42` | 랜덤 시드 |
| `--device` | `cuda` | 디바이스 (CUDA 없으면 자동 CPU fallback) |

### 출력 구조

```
spiral_test/output/<dataset>_<YYYYMMDD_HHMMSS>/
├── train_config.json           # 학습 설정
├── D2/
│   ├── proj_matrix.npy         # 투영 행렬 (D=2는 identity)
│   ├── classifier.pt           # 학습된 classifier
│   ├── diffusion_x.pt          # x-prediction diffusion model
│   ├── diffusion_e.pt          # e-prediction diffusion model
│   └── diffusion_v.pt          # v-prediction diffusion model
├── D8/
│   └── ...
├── D32/
│   └── ...
├── D128/
│   └── ...
└── D512/
    └── ...
```

### 모델 아키텍처

- **Classifier**: MLP (128→128→128→output), ReLU activation
- **Diffusion Model**: ResMLP with sinusoidal time embedding
  - Hidden dim: 256, Residual blocks: 5
  - 각 block: LayerNorm → Linear → ReLU → time projection → Linear → ReLU + skip connection

---

## 3. 추론 (`inference.py`)

학습된 모델을 로드하여 DSP guidance 기반 샘플링을 수행하고, 결과를 시각화합니다.

### DSP (Classifier) Guidance

```
v_guided = v_pred + s · ∇_z log p(y=target | x̂₀(z_t))
```

- `s` = guidance scale (0이면 unconditional)
- `x̂₀` = diffusion model이 예측한 clean data
- `p(y | x̂₀)` = classifier의 예측 확률

### 사용법

```bash
# 기본 (scale=0,2,5 / steps=50,100)
uv run python spiral_test/inference.py --exp doublespiral_20260113_123456

# guidance scale과 step 수 지정
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    -s 2.0 \
    -n 50

# 여러 scale × step 조합
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    -s 0.0 2.0 5.0 10.0 \
    -n 50 100 200

# target class 변경
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    --target_class 0

# 샘플 수 조정
uv run python spiral_test/inference.py \
    --exp doublespiral_20260113_123456 \
    --num_samples 5000
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--exp` | **(필수)** | `spiral_test/output/` 내 실험 폴더명 |
| `-s`, `--guidance_scale` | `0.0 2.0 5.0` | Guidance scale (복수 지정 가능) |
| `-n`, `--num_steps` | `50 100` | 샘플링 step 수 (복수 지정 가능) |
| `--num_samples` | `10000` | 생성할 샘플 수 |
| `--target_class` | `1` | Guidance 대상 클래스 |
| `--seed` | `42` | 랜덤 시드 |
| `--device` | `cuda` | 디바이스 |

### 파이프라인 단계

1. **데이터 & 모델 로드** — 2D 데이터 + 각 D별 투영행렬, classifier, diffusion model (x/e/v) 로드
2. **DSP guidance 샘플링** — 각 (guidance_scale, num_steps) 조합에 대해 Euler ODE 샘플링
3. **Metrics 계산** — 생성 샘플의 품질 정량 평가
4. **시각화** — 2D 재투영 + classifier decision boundary 배경

### Metrics

각 (D, prediction target) 조합에 대해 5개 metric을 계산합니다:

| Metric | 설명 |
|--------|------|
| `on_manifold_rate` | Manifold 위 샘플 비율 (`crossed_lines`, `half_arcs`만 지원, 나머지는 `null`) |
| `source_mmd` | 생성 샘플 ↔ 전체 GT 간 MMD (Gaussian kernel, median heuristic) |
| `target_mmd` | 생성 샘플 ↔ target class GT 간 MMD |
| `kl_div` | KL(p_gen ‖ p_target), dual KDE 기반 (singular covariance 시 `null`) |
| `class_accuracy` | Classifier가 target class로 분류한 비율 |

### 출력

```
<exp_dir>/results/
├── metrics_s0.0_steps50.json   # scale별, step별 정량 metrics
├── metrics_s2.0_steps100.json
├── s0.0_steps50.png            # scale별, step별 시각화 이미지
├── s2.0_steps100.png
├── ...
└── inference_config.json       # 추론 시 사용한 설정
```

각 이미지는 **(D값 × 4)** 그리드로 구성됩니다:

|  | Ground Truth | x-pred | e-pred | v-pred |
|--|--|--|--|--|
| **D=2** | 원본 데이터 | 생성 샘플 | 생성 샘플 | 생성 샘플 |
| **D=8** | 〃 | 〃 | 〃 | 〃 |
| **D=32** | 〃 | 〃 | 〃 | 〃 |
| **D=128** | 〃 | 〃 | 〃 | 〃 |
| **D=512** | 〃 | 〃 | 〃 | 〃 |

배경에는 classifier의 decision boundary가 색으로 표시됩니다.

---

## Quick Start

```bash
# 1. 데이터 생성
uv run python spiral_test/generate_data.py --name doublespiral

# 2. 모델 학습 (빠른 테스트: 작은 D, 적은 epoch)
uv run python spiral_test/train.py \
    --data doublespiral \
    --d_values 2 8 \
    --diffusion_epochs 100

# 3. 추론 (출력 폴더명 확인 후)
uv run python spiral_test/inference.py \
    --exp doublespiral_<YYYYMMDD_HHMMSS> \
    -s 0.0 2.0 5.0 \
    -n 50
```

---

## Notes

- **Self-contained**: 이 폴더의 코드는 메인 코드베이스(`src/jit_tfg/`)와 **완전히 독립적**입니다. 외부 모듈 import가 없으므로 메인 코드 변경의 영향을 받지 않습니다.
- **Dependencies**: `torch`, `numpy`, `matplotlib`, `scipy`, `tqdm` (모두 프로젝트의 기본 의존성에 포함)
- **GPU**: CUDA가 없으면 자동으로 CPU로 fallback됩니다.
- **재현성**: `--seed` 옵션으로 데이터 생성, 학습, 추론 모두 재현 가능합니다.
