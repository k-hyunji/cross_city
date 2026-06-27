# 0507 Solution Document — Cross-City Urban Region Representation

날짜: 2026-05-07
대상 모델: `DualModalNet` (sep_agg 변형)
학습 도시: NYC + Seoul → Singapore (zero-shot transfer)
현재 최고 결과: sep_agg_3, R² = -0.051 (log-space)

---

## 0. Provenance (데이터·아이디어 출처)

| 구성요소 | 출처 | 비고 |
|---|---|---|
| 아키텍처 (shared/spec split) | **SimMMDG** [^simmmdg] (NeurIPS 2023) | Multi-modal Domain Generalization 프레임워크 차용 |
| Satellite embedding [64] | **Google AlphaEarth Foundations** [^alphaearth] (2024) | 글로벌 foundation model 출력. 10m 해상도 다년간 sensor 통합 |
| POI embedding [64] (`morph_emb`) | **RegionContext** [^regioncontext] (knowledge-computing) | POI 패턴 임베딩, 도시별 학습 기반 |

이 출처 차이가 sat과 POI의 city-bias 성격을 다르게 만들고, 본 문서의 솔루션 우선순위에 직접 영향을 줌(아래 §4-A1 참조).

---

## 1. Current Model Explanation

### 1.1 파이프라인

```
Train (NYC + Seoul) → Extract embeddings → Linear probe (Ridge) → Singapore pop density
```

소스 도시에서 자기지도학습으로 region embedding [128]을 학습하고, target city에 대한 라벨 없이 zero-shot으로 전이.

### 1.2 아키텍처 (`DualModalNet`, `models/model.py`)

**입력**: 각 region에 속한 BG들의 satellite emb [M, 64], POI morph emb [M, 64], valid_mask [M].

**BG-level projection (4개)**:
```
sat_shared = Linear(64→32)   |   sat_spec = Linear(64→32)
poi_shared = Linear(64→32)   |   poi_spec = Linear(64→32)
```

**Region-level aggregation (`use_sep_agg=true`, 4개의 별도 aggregator)**:
```
sat_shared_region = AttnAgg(sat_shared, mask)   [32]
sat_spec_region   = AttnAgg(sat_spec,   mask)   [32]
poi_shared_region = AttnAgg(poi_shared, mask)   [32]
poi_spec_region   = AttnAgg(poi_spec,   mask)   [32]

region_shared = MLP(cat[sat_shared_region, poi_shared_region])  [32]   # CLIPSharedCon, CrossCity 입력
sat_region    = cat[sat_shared_region, sat_spec_region]         [64]
poi_region    = cat[poi_shared_region, poi_spec_region]         [64]
region_emb    = cat[sat_region, poi_region]                     [128]  # 다운스트림용
```

**AttentionAggregator**: self-attention → learnable pool query → FFN. transformer 스타일 set pooling.

**Frozen `sat_mean`**: dataset load 시 region별 `mean(sat_emb[valid BGs])` 계산 후 학습 내내 고정. Cross-city positive mining의 anchor로 사용됨 (학습 중 변하지 않는 visual fingerprint).

### 1.3 손실 함수 (`models/losses.py`)

```
L_total = λ_contrast · L_clip_shared
        + λ_dis      · L_dis
        + λ_proto    · L_proto
        + λ_cross    · L_cross_city
        + λ_vicreg   · L_vicreg
        (+ λ_adv     · L_adv  — DANN, 옵션)
```

| Loss | 작동 위치 | 역할 |
|---|---|---|
| `CLIPSharedCon` | sat_shared_region[32] × poi_shared_region[32] | sat-poi 같은 region을 가깝게. spec branch에 grad 누출 없음 |
| `BGDisLoss` | BG 단위 \|cos(spec, shared.detach())\| | shared/spec subspace 직교화 |
| `FunctionalPrototypeLoss` | region_emb[128] vs K=16 prototypes (Sinkhorn) | functional archetype 학습 (의도) |
| `CrossCitySharedLoss` | region_shared[32], InfoNCE | sat_mean 기준 cross-city positive 매칭 |
| `WithinCityVarLoss` (VICReg variance term) | region_emb[128] (도시 평균 제거 후) | within-city std ≥ γ 강제, 표현 붕괴 방지 |

### 1.4 학습 설정

- Adam (lr=1e-4, wd=1e-4), CosineAnnealingLR, 100 epoch
- `balanced_sampler=true`: 16 NYC + 16 Seoul / batch
- Train/val 80/20, gradient clipping max_norm=1.0
- Linear probe: log1p(pop_density) → Ridge (α grid search by source train R²) → de-log1p로 MAE/RMSE, log-space로 R²

---

## 2. Strengths

### 2.1 아키텍처 측면

1. **Shared/Spec subspace 분리 (SimMMDG-derived)** — modality-invariant 정보(`shared`)와 modality-specific 정보(`spec`)를 별도 32-d 공간으로 분리. cross-modal contrastive loss(CLIPSharedCon)가 shared subspace에만 작용하므로 spec branch가 cross-modal "평균화 압력"을 받지 않음. SimMMDG에서 영상-오디오 modal에 적용된 구성을 sat-POI에 그대로 차용. 단일 64-d 공간에서 학습하는 baseline(CLIPRegionCon) 대비 region별 다양성 보존에 유리.

2. **Per-modality, per-subspace aggregator (4개)** — 각 (sat/poi × shared/spec) 조합이 독자적인 attention pooling을 학습. set 단위 입력에서 branch별 의미 보존.

3. **Frozen `sat_mean` anchor (AlphaEarth backbone)** — AlphaEarth foundation model 출력의 region 평균을 학습 내내 고정. 글로벌 다중 센서 통합으로 학습된 임베딩이라 cross-city positive mining 기준으로 적합. 학습된 표현이 자기 참조적으로 무너지는 것(self-collapse)을 방지하는 외부 anchor 역할도 함.

4. **VICReg variance term** — within-city 평균 제거 후 std ≥ γ 강제. cross-city alignment loss가 모든 도시를 한 점으로 끌어당기는 것을 막아주는 안전장치.

### 2.2 학습 측면

1. **모듈식 loss switching** — config의 `use_*` 플래그로 각 loss를 on/off. 본 프로젝트의 ablation이 가능했던 핵심 인프라.

2. **Balanced batch sampling** — 도시 간 region 수 차이(NYC 2312 / Seoul 426)가 6배지만 batch 단위로 16:16 균등화 → loss가 큰 도시에 휩쓸리지 않음.

3. **log1p target + Ridge α grid search** — Singapore 분포(std=10,611, log1p std=3.23)가 소스보다 3.7× 넓은 상황에서 log1p 변환으로 회귀 목표 분포를 정규화.

### 2.3 진단/실험 인프라

1. **분석 자동화 (`analyze/`)** — t-SNE, K-means ARI, within/between city sim 분포, sat_sim threshold 실효성 검증까지 14개의 분석 문서가 누적되어 모델 진단의 근거 기반이 명확.

2. **세분화된 ablation** — sep_agg_1~7 + Option A/B를 구조화된 비교표로 추적 (`analyze_14.txt`).

---

## 3. Weaknesses

### 3.1 구조적 약점 — SimMMDG 대비 빠진 컴포넌트

원본 SimMMDG의 4가지 핵심 구성: ① shared/spec split, ② **supervised contrastive on shared (with class labels)**, ③ **distance constraint to promote diversity in specific features**, ④ **cross-modal translation module**. 본 코드는 ① 만 완전 차용, 나머지는 부분/누락:

| SimMMDG 컴포넌트 | 본 코드 대응 | 상태 |
|---|---|---|
| ① Shared/Spec split | `*_shared_proj`, `*_spec_proj` | ✅ 완전 |
| ② Supervised contrastive on shared | `CLIPSharedCon` (cross-modal 자기지도) | ⚠️ 변형. zero-shot transfer라 class label 없어 cross-modal로 대체. **그러나 cross-modal 정렬 ≠ 도메인 불변성** |
| ③ Distance constraint on spec features | `BGDisLoss` (`\|cos(spec, shared.detach())\|`) | ⚠️ 부분. 직교성만 강제, **diversity 보장 없음**. 또한 BG 단위 → region 단위 누수 |
| ④ Cross-modal translation module | 없음 | 🔴 누락. sat → poi, poi → sat 재구성으로 mutual information 강화하는 SimMMDG의 핵심 정규화 기제 부재 |

이 갭이 본 모델의 약점을 만들어냄:

1. **Region-level disentanglement 부재** — `region_shared`는 MLP fusion으로 만들어지지만, `sat_spec_region`, `poi_spec_region`과의 직교성을 강제하는 region-level loss가 없음. SimMMDG의 distance constraint가 BG-level에만 약하게 적용됨.

2. **`BGDisLoss` gradient <1%** — λ=0.3 × 평균 ~0.05 = 0.015. 거의 cosmetic. SimMMDG에서는 이 항이 표현 다양성의 핵심 제약이지만 본 모델에서는 무력.

3. **`region_emb[128] = cat[sat[64], poi[64]]`** — 다운스트림에서 사용되는 최종 임베딩은 단순 concat. 모달리티 간 상호작용이 fusion layer 없이 학습되지 않음. SimMMDG의 cross-modal translation이 있다면 sat과 poi가 서로의 정보를 공유한 표현이 자연스럽게 형성됐을 것.

4. **POI centroid [2] 미사용** — `poi_emb.npy`에 BG 좌표가 있지만 모델 입력으로 들어가지 않음. spatial context 신호 누락.

5. **`SatAlignLoss`(legacy)와 `CrossCitySharedLoss`의 중복** — 두 loss 모두 sat_mean 기반 positive mining을 하지만 서로 다른 region 위에서 동작. 코드 잔재로 인한 혼란.

6. **Domain Generalization으로서의 본질적 mismatch** — SimMMDG는 **여러 도메인(소스)에서 학습 → 단일 unseen 도메인(타깃) 일반화**를 하는데, 도메인 라벨을 사용한 supervised contrastive가 핵심. 본 프로젝트는 도메인=도시이고 도시는 식별되지만 region별 class label이 없는 자기지도 setting이라, SimMMDG의 핵심 supervised signal이 빠진 상태에서 자기지도(cross-modal)로 대체됨. 이 대체가 충분한지에 대한 근거가 없음.

### 3.2 학습 신호의 약점

1. **Cross-city positive mining 신호 노이즈** — sat_sim threshold=0.4 기준 cross-city positive ~14.9% vs within-city pair >0.4가 22–24% (analyze_11). within-city 쌍이 cross-city positive보다 sat_sim이 더 높은 경우가 흔함 → InfoNCE denominator에 들어가면 모순 gradient 발생.

2. **POI taxonomy 불일치 (Problem 7)** — NYC PLUTO / Seoul 한국 코드 / Singapore URA 카테고리가 서로 다른 분류 체계. raw POI cross-city sim의 max=0.271, mean=0.092. 어떤 threshold도 의미 있는 positive를 만들 수 없음. POI가 cross-city alignment에 기여 못함.

3. **Pre-trained feature의 city bias (Problem 6, root cause)** — raw sat과 raw POI 모두 K=3 ARI = 0.87/0.94로 도시 식별 정보를 강하게 인코딩. PCA에서 POI PC1이 분산의 65.3%를 설명 (사실상 도시축). 이 위에 어떤 contrastive loss를 쌓아도 도시축을 dropping 시키는 명시적 기제가 없으면 학습된 임베딩도 도시 분리됨.

4. **`FunctionalPrototypeLoss`의 비기능성** — K=16에서 raw feature K-means AvgCityPurity=0.999. Sinkhorn 균등 할당이 prototype을 "city sub-region"으로 만들 뿐 "cross-city functional archetype"이 되지 않음. val_proto loss가 100 epoch 내내 2.67→2.63으로 거의 평탄.

5. **`CrossCitySharedLoss`의 미수렴** — val_cc 3.29 → 3.04 (목표 1.0~2.0). soft mining이 충분히 의미 있는 positive를 못 찾고 있거나, λ=0.1 스케일로 VICReg에 묻힘.

### 3.3 실험적으로 확인된 약점

1. **R² always negative** — sep_agg_3의 R²=-0.051이 최고이지만 여전히 음수. mean prediction보다 못함. 어떤 ablation에서도 R² > 0 달성 못함.

2. **Option A/B 모두 sep_agg_3보다 나쁨** —
   - Option A (CC loss off): R²=-0.461 (최저)
   - Option B (denom=cross_only): R²=-0.360
   - sep_agg_3 (full denom, λ=0.1): R²=-0.051
   - 즉 within-city 쌍을 denominator에서 제거하면 도리어 도시 클러스터링 발생.

3. **Singapore 분포 이질성 (Problem 5)** — log1p std: NYC 0.879, Seoul 0.740, Singapore **3.229**. Singapore에는 MRT/공항/공공기관(거주인구≈0)과 고밀도 주거가 함께 있는 bimodal 분포. 표현이 완벽해도 R² 상한이 구조적으로 낮음.

---

## 4. Problems (확정 진단)

이미 문서화된 8개 문제(`problems.md`) 중 본 프로젝트의 핵심 차단 요인은 다음 세 가지:

### Problem A — Pre-trained Feature City Bias (sat/POI 성격이 다름)

**증거**: K=3 ARI = 0.866 (sat) / 0.943 (POI). 즉 raw input만으로 도시 분류가 87–94% 정확.

**핵심 통찰 — 두 모달리티의 city bias 성격이 본질적으로 다름:**

| 모달리티 | 인코더 | city bias 성격 | 처리 방향 |
|---|---|---|---|
| Satellite | **AlphaEarth** (글로벌 foundation model) | **실제 신호**. NYC/Seoul/Singapore가 정말로 다르게 생김. 인코더가 도시-특정 학습된 게 아니라 글로벌 기준에서 정확히 표현한 결과 | 제거 대상 아님. spec subspace로 보존하고, shared subspace에서만 도시 불변성 강제 |
| POI | **RegionContext** (도시별 기반 학습) | **인공물에 가까움**. 분류 체계(PLUTO/한국/URA)와 인코더 학습 데이터가 도시별로 분리됨. K=3 ARI=0.943은 "공간이 다른 게 아니라 임베딩 공간 자체가 다른" 결과 | 적극 제거. 텍스트 기반 LFM 임베딩으로 교체 또는 강한 alignment 필요 |

**기존 진단의 수정**: 이전 problems.md는 sat과 POI 모두 "encoder를 갈아끼우자"고 했지만, **sat은 이미 foundation model이라 교체 효과 미미**. AlphaEarth가 잡아낸 도시 외관 차이는 진짜 신호이고, 본 모델은 그것을 spec branch에 잘 보존하는 방향으로 가야 함. 반면 **POI는 인코더가 진짜로 city-specific**이라 교체 시 큰 이득 가능.

**영향**: city-mean subtraction은 평균만 제거할 뿐 manifold 형태(NYC=S자, Seoul=덩어리, Singapore=가는 선)는 그대로 유지(analyze_12). 이 형태 차이가 sat에서는 의미 있는 신호일 수 있지만 POI에서는 noise.

### Problem B — Cross-City Positive Mining의 본질적 한계

**증거 (analyze_11)**: city-mean subtraction 후 sat_sim 분포에서 within-city pair >0.4가 24%, cross-city pair >0.4가 15%. 두 분포가 거의 완전히 겹침. **어떤 threshold로도 within/cross 통계적 구분 불가**.

**영향**: `CrossCitySharedLoss`가 "외관 유사도"를 매칭할 뿐 "기능적 동등성"을 보장하지 못함. 학습된 cross-city pair는 절반 이상이 false positive 가능성.

### Problem C — Within-City Repulsion과 Cross-City Pull의 본질적 모순

**증거 (sep_agg_3 vs Option A vs Option B)**:
- CC loss 끄면 (Option A) → within-city 다양성 회복 기대했으나 R²=-0.461 최저
- denom=cross_only (Option B) → cross-city signal만 살림. R²=-0.360
- full denom (sep_agg_3) → 모순 gradient 가지지만 R²=-0.051 최고

**해석**: full in-batch denominator의 within-city negative가 의도치 않게 within-city repulsion 역할(VICReg 보완)을 함. 이걸 빼면 city-domain clustering 발생. 즉 **모순 gradient가 사실은 약한 cross-city pull + 강한 within-city push의 균형점**이었음. 이 균형은 운 좋게 맞은 것이지 설계된 것이 아님.

---

## 5. Solutions (recent research 기반)

### Solution 0 — SimMMDG의 빠진 컴포넌트 복원 (최우선)

**문제와의 연결**: §3.1 (SimMMDG 갭), Problem C (모순 gradient의 임시 균형)

**근거 — SimMMDG (NeurIPS 2023)** [^simmmdg]: 본 코드의 아키텍처 원본. 빠진 두 컴포넌트:
- **Distance constraint on specific features**: spec branch가 단순 직교가 아니라 **다양성 자체를 갖도록** 강제. SimMMDG는 same-class 내에서 distance를 키우는 형태로 적용.
- **Cross-modal translation**: sat→poi, poi→sat 재구성 헤드. shared가 정말로 두 모달리티 공통 정보를 담도록 강제하는 자기지도 정규화.

**적용 안**:

(a) **Spec diversity loss 추가**: BG 단위 또는 region 단위에서
```
L_spec_div = -mean( pairwise_distance(sat_spec_region) ) - mean( pairwise_distance(poi_spec_region) )
```
또는 SimMMDG 원본대로 within-class far / between-class close 형태로 (본 setting에서 "class"가 없으니 cluster pseudo-label로 대체 — 예: K-means로 할당).

(b) **Cross-modal translation head**:
```
sat_shared → MLP_s2p → poi_shared_pred  (regression to true poi_shared)
poi_shared → MLP_p2s → sat_shared_pred  (regression to true sat_shared)
L_trans = MSE(sat_shared_pred, sat_shared.detach()) + MSE(poi_shared_pred, poi_shared.detach())
```

**기대 효과**:
- spec branch가 의미 있는 modality-specific 정보를 담게 됨 (현재는 cosmetic)
- shared가 "양 모달리티에서 재구성 가능한 정보"로 명확히 정의됨 → CLIPSharedCon보다 강한 정렬 신호
- Problem C(모순 gradient의 운 좋은 균형)를 명시적 정규화로 대체

### Solution 1 — POI를 cross-city signal에서 분리, 의미 정렬은 LFM으로 위임

**문제와의 연결**: Problem A, B, 3.2-2 (POI taxonomy 불일치)

**근거 — ReFound (KDD 2024)** [^refound]: POI는 "이름 텍스트"로 받아 LFM(language foundation model)으로 임베딩. raw POI 카테고리 코드(PLUTO/한국코드/URA)를 직접 쓰지 않고 자연어 설명으로 정규화함으로써 도시 간 taxonomy 갭 해소. 그리고 LFM/VFM/VLFM 세 teacher로부터 knowledge distillation을 받아 학생 모델이 domain-invariant 표현 획득.

**적용 안**:
- 각 BG의 POI 리스트를 "Korean restaurant, subway entrance, park"처럼 자연어 텍스트화
- pretrained LFM (e.g., E5, BGE)으로 임베딩 → 64-d로 투영
- 현재 `morph_emb`를 이 LFM 기반 POI 임베딩으로 교체

**기대 효과**: POI cross-city sim의 raw max=0.271 한계 해소. POI가 비로소 cross-city positive mining의 부신호로 작동 가능.

### Solution 2 — Cross-City Positive Mining을 grid-cell 단위로 fine-grained화

**문제와의 연결**: Problem B (region-level positive mining의 noise)

**근거 — UrbanVerse (ICLR 2026)** [^urbanverse]: city를 fine-grained grid cell로 분할하고 cell 단위 임베딩을 학습. region 임베딩은 자신과 겹치는 cell의 aggregation. cell은 region보다 작아 외관-기능 매칭이 sharp하고 cross-city overlap이 더 잘 일어남. NYC/Chicago/SF 6개 task에서 SOTA 대비 최대 35.89% 향상.

**적용 안**:
- 현재 BG 단위가 이미 fine-grained하지만, BG 단위 cross-city sat_sim은 측정/사용되지 않음
- BG 단위 sat_mean에서 cross-city positive를 mining하고, region-level loss는 그 결과를 위로 propagate
- 또는 BG → cell → region의 3-level aggregation 도입

**기대 효과**: region-level에서 24%가 within > cross인 신호 노이즈를, BG-level에서 더 정밀한 매칭으로 줄임.

### Solution 3 — Domain Adversarial Training으로 city axis 명시적 제거

**문제와의 연결**: Problem A (city bias root cause)

**근거 — DANN (Ganin et al., JMLR 2016)** [^dann]: gradient reversal layer로 도메인 분류기를 학습 시 latent에서 도메인 정보를 빼는 방향으로 backbone을 학습. 본 코드에 이미 `use_adv` 플래그로 GradReverse + city_classifier가 구현되어 있으나 비활성. 최근 연구들은 contrastive 학습과 결합 시 false-negative 처리에 주의해야 한다고 제안.

**적용 안**:
- `use_adv: true`로 켜고 `lambda_adv` sweep (0.05, 0.1, 0.2)
- city_classifier는 `region_shared[32]`에 작용. spec branch는 의도적으로 city 정보 유지(modality-specific signal로 사용)
- DANN 만으로는 over-shooting 위험 → CLIPSharedCon + VICReg와 병행

**기대 효과**: shared subspace의 K=3 city-ARI를 직접 낮춤. analyze_2의 PCA에서 POI PC1=65.3%로 보이는 도시축 제거에 직접적.

### Solution 4 — POI 인코더만 city-agnostic으로 교체 (sat은 그대로)

**문제와의 연결**: Problem A (POI city bias = 인코더 인공물)

**상황 정리**: sat 임베딩은 이미 **AlphaEarth foundation model** [^alphaearth] 출력 — 글로벌 학습된 backbone이라 추가 교체 시 한계 효용 적음. K=3 ARI=0.866은 인코더 결함이 아니라 NYC/Seoul/Singapore가 정말로 다르게 생긴 결과. 반면 POI는 RegionContext 기반으로 도시별 학습된 가능성이 높아 K=3 ARI=0.943은 진짜 인공물.

**근거 — ReFound (KDD 2024)** [^refound]: POI를 자연어로 변환 후 LFM 임베딩 사용 → 도시별 분류 코드 차이를 LFM의 의미 공간에서 자동 정렬. 또한 **UrbanCLIP** [^urbanclip], **UrbanVerse** [^urbanverse]도 cross-city transfer에서 SOTA.

**적용 안**:
- 현재 `morph_emb` (RegionContext 출력) 그대로 두는 경우와, LFM 텍스트 임베딩 (Solution 1)으로 교체하는 경우 ablation
- 또는 두 임베딩을 concat한 hybrid POI 표현
- sat 측은 AlphaEarth 그대로 유지 (교체 효용 없음)

**기대 효과**: POI K=3 ARI 0.94 → 0.6 이하 기대. sat ARI(0.87)는 그대로 유지되지만 그건 진짜 신호이므로 spec branch에 보존되는 게 정상.

### Solution 5 — VICReg을 region_shared subspace까지 확장

**문제와의 연결**: 3.1-2 (region-level shared/spec 직교화 부재)

**근거 — VICReg (ICLR 2022)** [^vicreg]: variance + invariance + covariance 세 항으로 collapse를 방지. 본 코드는 variance term만 사용. covariance term은 임베딩 차원들이 redundant하게 같은 정보를 담는 것을 막음.

**적용 안**:
- 현재 `WithinCityVarLoss`는 variance term만 적용 (region_emb[128]).
- covariance term 추가: off-diagonal cov 제곱합을 페널라이즈 → 차원별 독립성
- region_shared[32]에도 별도 VICReg 적용해서 shared subspace 차원 redundancy 방지

**기대 효과**: K=16 prototype이 city sub-region이 아닌 진짜 functional axis를 학습할 여지 확보.

### Solution 6 — Hard/False Negative Curriculum Mining

**문제와의 연결**: Problem C (모순 gradient)

**근거 — Curricular Negative Weighting (2024)** [^curr_neg]: easy/hard negative를 curriculum으로 가중. 가장 어려운 0.1% negative는 false negative일 가능성이 높아 오히려 감점. 학습 후반에는 hard negative에 더 가중하되 false negative 의심 항에는 L2 정규화.

**적용 안**:
- `CrossCitySharedLoss`에서 within-city pair의 sat_sim이 cross-city positive보다 높은 경우(false negative 후보)를 식별 → denominator에서 가중치 감소
- 또는 epoch에 따라 sat_threshold를 0.3 → 0.5로 점진 상향 (curriculum)

**기대 효과**: Option B에서 잃었던 within-city repulsion을 부분적으로 회복하면서, 가장 노이즈가 큰 within-city negative만 선택적으로 약화.

### Solution 7 — Singapore 분포 mismatch에 대한 robust evaluation

**문제와의 연결**: 3.3-3 (Singapore bimodal pop, log1p std=3.23)

**근거**: Singapore에는 거주인구 거의 0인 공공/교통/기관 region이 32.7% 존재 (analyze_3). pop density 단일 metric으로는 표현 품질을 정확히 측정 못함.

**적용 안**:
- `data/singapore/landuse_gt_list.csv`를 활용한 land-use classification probe 추가
- pop density는 보조 metric으로 두고 multi-task로 평가
- pop 자체도 log1p가 아니라 percentile rank로 변환 후 Spearman ρ 측정

**기대 효과**: 표현 품질 개선이 metric에 반영되도록 평가 신호의 noise 감소.

---

## 6. 실행 우선순위 권장

| 순위 | 솔루션 | 비용 | 기대 효과 | 의존성 |
|---|---|---|---|---|
| 1 | **Solution 0**: SimMMDG 빠진 컴포넌트 (cross-modal translation + spec diversity) | 중간 | 아키텍처 원본의 의도 회복, Problem C 직접 대응 | losses.py + model.py 수정 |
| 2 | Solution 7: 평가 metric 확장 (land use probe) | 낮음 | 신호/잡음비 개선 | 즉시 가능 |
| 3 | Solution 3: DANN 활성화 (`use_adv=true`) | 낮음 | shared subspace city-ARI 직접 감소 | config 변경만 |
| 4 | Solution 5: VICReg covariance term 추가 | 중간 | prototype 의미화 | 코드 수정 |
| 5 | Solution 6: false negative 인식 가중 mining | 중간 | Problem C 직접 대응 | losses.py 수정 |
| 6 | Solution 1+4: POI를 LFM 임베딩으로 교체 | 높음 | POI city bias(인공물) 해소 | 데이터 재처리 |
| — | ~~sat backbone 교체~~ | — | AlphaEarth가 이미 foundation model이라 효용 거의 없음 | — |

**전략 권장**:

- **단기 (sep_agg_8~9)**: **Solution 0** + Solution 3. SimMMDG 원본 아이디어를 충실히 구현 + DANN 활성화. 코드 수정만으로 가능하며 본 모델의 정체성(SimMMDG-기반)을 명확히 함.
- **중기 (sep_agg_10+)**: Solution 7로 평가 강화, Solution 5/6로 표현 품질 미세 조정.
- **장기**: Solution 1+4로 POI 인코더를 LFM 기반으로 전환. 데이터 재처리 비용은 크지만 root cause 중 가장 해결 가능한 부분.

---

## 참고 문헌

[^simmmdg]: Dong, Nejjar, et al. *SimMMDG: A Simple and Effective Framework for Multi-modal Domain Generalization*. NeurIPS 2023. [arXiv:2310.19795](https://arxiv.org/abs/2310.19795) · [Code](https://github.com/donghao51/SimMMDG)
[^alphaearth]: Google DeepMind. *AlphaEarth Foundations: An Embedding Field Model for Accurate and Efficient Global Mapping from Sparse Label Data*. 2024. [arXiv:2507.22291](https://arxiv.org/abs/2507.22291) · [Earth Engine Catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
[^regioncontext]: Knowledge Computing Lab. *RegionContext*. [Code](https://github.com/knowledge-computing/regioncontext)
[^refound]: Zhou et al. *ReFound: Crafting a Foundation Model for Urban Region Understanding upon Language and Visual Foundations*. KDD 2024. [PDF](http://zhoujingbo.github.io/paper/2024ReFoundKDD.pdf)
[^urbanclip]: Zhang et al. *UrbanCLIP: Learning Text-enhanced Urban Region Profiling with Contrastive Language-Image Pretraining from the Web*. WWW 2024. [Code](https://github.com/siruzhong/WWW24-UrbanCLIP)
[^urbanverse]: *UrbanVerse: Learning Urban Region Representation Across Cities and Tasks*. ICLR 2026. [arXiv:2602.15750](https://arxiv.org/abs/2602.15750)
[^dann]: Ganin et al. *Domain-Adversarial Training of Neural Networks*. JMLR 2016. [arXiv:1505.07818](https://arxiv.org/abs/1505.07818)
[^vicreg]: Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
[^curr_neg]: *Mining negative samples on contrastive learning via curricular weighting strategy*. Information Sciences 2024.
