# FastFace v5 Architecture Documentation

## Overview

FastFace v5는 **스타일 이미지 기반 얼굴 생성** 시스템입니다. 사용자가 제공한 얼굴 이미지의 정체성(identity)을 보존하면서, 스타일 이미지의 배경, 구도, 색감을 반영한 새로운 이미지를 생성합니다.

### 핵심 목표

```
[Face Image] + [Style Image] + [Text Prompt] → [Generated Image]
     ↓               ↓              ↓
  얼굴 정체성     배경/구도/색감    추가 수정 지시
```

---

## System Architecture

### 전체 구조도

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastFace v5 Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  Face Image  │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ├──────────────────┬─────────────────┐                      │
│         ▼                  ▼                 ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ InsightFace  │  │  CLIP ViT    │  │  Face Crop   │              │
│  │ (buffalo_l)  │  │ (224x224)    │  │  (norm_crop) │              │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘              │
│         │                 │                                         │
│         ▼                 ▼                                         │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │ Face ID      │  │ Face CLIP    │                                │
│  │ Embedding    │  │ Embedding    │                                │
│  │ (512-dim)    │  │ (257, 1280)  │                                │
│  └──────┬───────┘  └──────┬───────┘                                │
│         │                 │                                         │
│         │                 │  ┌──────────────┐                       │
│         │                 │  │ Style Image  │                       │
│         │                 │  └──────┬───────┘                       │
│         │                 │         │                               │
│         │                 │         ├────────────┬────────────┐     │
│         │                 │         ▼            ▼            ▼     │
│         │                 │  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│         │                 │  │CLIP ViT  │ │   VAE    │ │  MiDaS   ││
│         │                 │  │(224x224) │ │ Encoder  │ │  Depth   ││
│         │                 │  └────┬─────┘ └────┬─────┘ └────┬─────┘│
│         │                 │       │            │            │       │
│         │                 │       ▼            ▼            ▼       │
│         │                 │  ┌──────────┐ ┌──────────┐ ┌──────────┐│
│         │                 │  │Style CLIP│ │  Init    │ │  Depth   ││
│         │                 │  │Embedding │ │ Latents  │ │   Map    ││
│         │                 │  └────┬─────┘ └────┬─────┘ └────┬─────┘│
│         │                 │       │            │            │       │
│         │                 ▼       ▼            │            │       │
│         │          ┌─────────────────────┐    │            │       │
│         │          │   CLIP Blending     │    │            │       │
│         │          │ (1-α)·face + α·style│    │            │       │
│         │          └──────────┬──────────┘    │            │       │
│         │                     │               │            │       │
│         ▼                     ▼               ▼            ▼       │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                    SDXL UNet                                   ││
│  │  ┌────────────────────────────────────────────────────────┐   ││
│  │  │               Cross-Attention Layers                    │   ││
│  │  │                                                         │   ││
│  │  │  [Text Prompt] ──► Text Attention                       │   ││
│  │  │                         │                               │   ││
│  │  │  [FaceID Embed] ──► IP-Adapter FaceID ──► + ◄── DCG    │   ││
│  │  │                                            │             │   ││
│  │  │  [Blended CLIP] ──► CLIP Shortcut ─────────┘             │   ││
│  │  │                                                         │   ││
│  │  └────────────────────────────────────────────────────────┘   ││
│  │                          │                                     ││
│  │  [Init Latents] ──► img2img (denoising_strength)              ││
│  │                          │                                     ││
│  │  [Depth Map] ──► ControlNet ──► Structure Conditioning         ││
│  │                          │                                     ││
│  └──────────────────────────┼─────────────────────────────────────┘│
│                             ▼                                      │
│                    ┌──────────────┐                                │
│                    │   VAE Decode │                                │
│                    └──────┬───────┘                                │
│                           ▼                                        │
│                   [Generated Image]                                │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Face Processing

#### InsightFace (buffalo_l)
- **용도**: 얼굴 검출 및 identity embedding 추출
- **출력**: 512차원 normed_embedding
- **특징**: 얼굴 방향, 표정에 robust한 identity 특징

#### Face CLIP Embedding
- **용도**: 얼굴의 시각적 특징 추출 (CLIP 공간)
- **출력**: (257, 1280) hidden states
- **입력**: 224x224로 crop된 얼굴 이미지

### 2. Style Processing

#### Style CLIP Embedding
- **용도**: 스타일 이미지의 색감, 분위기, 의미 추출
- **출력**: (257, 1280) hidden states
- **입력**: 224x224로 리사이즈된 스타일 이미지

#### VAE Latent Encoding (img2img)
- **용도**: 스타일 이미지를 latent space로 인코딩
- **출력**: (4, H/8, W/8) latent tensor
- **역할**: 생성 시작점으로 사용 (denoising_strength로 제어)

#### MiDaS Depth Estimation (ControlNet)
- **용도**: 스타일 이미지의 구조/깊이 정보 추출
- **출력**: Depth map (grayscale image)
- **역할**: ControlNet을 통해 공간 구조 유지

### 3. CLIP Blending (v4 핵심)

```python
blended = (1 - style_strength) * face_clip + style_strength * style_clip
```

| style_strength | 얼굴 비중 | 스타일 비중 | 효과 |
|---------------|----------|------------|------|
| 0.0 | 100% | 0% | 원본 얼굴만 반영 |
| 0.3 | 70% | 30% | 권장값 - 균형 잡힌 스타일 적용 |
| 0.5 | 50% | 50% | 강한 스타일 전이 |
| 0.7 | 30% | 70% | 매우 강한 스타일 (identity 약화 가능) |

### 4. Decoupled Guidance (DCG)

DCG는 텍스트 guidance와 이미지 guidance를 분리하여 더 정밀한 제어를 가능하게 합니다.

| DCG Type | Batch Size | 구성 |
|----------|-----------|------|
| 3 (기본) | 3 | [negative, positive_text, positive_image] |
| 4 | 4 | [negative, positive_text, positive_image, positive_both] |

---

## Generation Modes

### Mode A: Style Image 있음 (v5 주요 모드)

```
Style Image Present
        │
        ├──► img2img 활성화 (denoising_strength < 1.0)
        │    └─► 스타일 이미지 latent에서 시작
        │
        ├──► ControlNet 활성화 (if loaded)
        │    └─► 스타일 이미지 depth로 구조 유지
        │
        ├──► CLIP Blending 활성화
        │    └─► face_clip + style_clip 블렌딩
        │
        └──► FaceID 활성화
             └─► 얼굴 정체성 보존
```

**파라미터:**
- `style_strength`: CLIP 블렌딩 비율 (0.0-1.0)
- `denoising_strength`: img2img 강도 (0.2-1.0)
  - 낮을수록 스타일 이미지 보존
  - 높을수록 텍스트/FaceID 영향 증가
- `ip_adapter_scale`: FaceID 강도 (0.0-1.0)

### Mode B: Style Image 없음 (FaceID Only)

```
No Style Image
        │
        ├──► img2img 비활성화
        │    └─► 순수 노이즈에서 시작
        │
        ├──► ControlNet 비활성화
        │
        ├──► CLIP Blending 비활성화
        │    └─► face_clip만 사용
        │
        └──► FaceID 활성화
             └─► 텍스트 프롬프트로 구도 지정
```

---

## Known Issues and Limitations

### Issue #2: 스타일 이미지에 인물 없을 때 품질 저하

**문제 상황:**
- 스타일 이미지: 앱 스크린샷, 풍경, 배경 등 (인물 없음)
- 결과: 흐릿하고 불완전한 인물 생성

**원인 분석:**

```
스타일 이미지에 인물 없음
        │
        ├──► img2img latents: "인물 형태 없는" latent에서 시작
        │
        ├──► ControlNet depth: "인물 실루엣 없는" 구조 강제
        │
        ├──► Style CLIP: "앱 화면", "배경" 등 semantic 인코딩
        │
        └──► FaceID: "이 얼굴을 생성해"
             │
             └──► 충돌! 4개 신호 중 3개가 "인물 없음"
```

**신호 충돌 테이블:**

| Component | 인물 있는 스타일 | 인물 없는 스타일 |
|-----------|---------------|---------------|
| img2img latents | 인물 형태 포함 | 인물 형태 없음 |
| ControlNet depth | 인물 실루엣 있음 | 인물 실루엣 없음 |
| Style CLIP | "사람이 있는 장면" | "사람이 없는 장면" |
| FaceID | 얼굴 생성 지시 | 얼굴 생성 지시 |
| **충돌** | 없음 (일관된 신호) | **3:1 충돌** |

**제안 해결책: 조건부 파이프라인**

```python
def has_person_in_style(style_image):
    """스타일 이미지에 인물이 있는지 감지"""
    faces = face_analyzer.get(style_image)
    return len(faces) > 0

def execute(face_image, style_image, ...):
    if style_image is not None:
        if has_person_in_style(style_image):
            # 현재 방식 유지
            # img2img + ControlNet + CLIP Blending + FaceID
            use_img2img = True
            use_controlnet = True
        else:
            # 인물 없는 스타일: 충돌 방지
            # CLIP만 사용 (색감/분위기), img2img/ControlNet 비활성화
            use_img2img = False
            use_controlnet = False
            # 텍스트 프롬프트로 구도 지정
```

---

## Parameter Reference

### Frontend Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `ips` (ip_adapter_scale) | 0.8 | 0.0-1.0 | FaceID 강도 |
| `style_strength` | 0.3 | 0.0-1.0 | CLIP 블렌딩 스타일 비율 |
| `denoising_strength` | 0.6 | 0.2-1.0 | img2img 강도 (낮을수록 스타일 보존) |
| `inference_steps` | 4 | 1-50 | 생성 스텝 수 |
| `lora_scale` | 0.6 | 0.0-1.0 | LoRA 강도 (Hyper-SDXL) |
| `seed` | 42 | any int | 랜덤 시드 |

### Backend Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `controlnet_conditioning_scale` | 0.7 | ControlNet 영향력 |
| `dcg_type` | 3 | DCG 배치 타입 |

### Model Configuration

| Model | Path | Purpose |
|-------|------|---------|
| Base Model | `SG161222/RealVisXL_V5.0` | SDXL 기반 실사 모델 |
| IP-Adapter | `ip-adapter-faceid-plusv2_sdxl.bin` | 얼굴 정체성 보존 |
| ControlNet | `diffusers/controlnet-depth-sdxl-1.0` | 구조 보존 |
| Depth Estimator | MiDaS (via controlnet-aux) | 깊이 추출 |

---

## File Structure

```
FastFace/
├── src/
│   ├── sdxl_custom_pipeline.py    # 핵심 파이프라인 (DCG + IP-Adapter + ControlNet)
│   ├── controlnet_pipeline.py     # ControlNet 통합 유틸리티
│   ├── dcg.py                     # Decoupled Guidance 구현
│   └── utils.py                   # 헬퍼 함수
├── backend/
│   └── main.py                    # FastAPI 서버
├── frontend/
│   ├── app/page.tsx               # Next.js UI
│   └── lib/api.ts                 # API 클라이언트
└── docs/
    ├── DEVELOPMENT_HISTORY.md     # 버전별 개발 히스토리
    └── V5_ARCHITECTURE.md         # 이 문서
```

---

## Execution Flow

### 1. Image Upload
```
Client ──► POST /upload ──► Backend saves to uploads/
                                    │
                                    ▼
                            Return file_id, url
```

### 2. Generation Request
```
Client ──► POST /generate ──► Backend creates Task
                                    │
                                    ▼
                            Background thread starts
                                    │
                                    ▼
                            Load model (if needed)
                                    │
                                    ▼
                            pipe.execute(...)
                                    │
                                    ▼
                            Save output to outputs/
                                    │
                                    ▼
                            Update task status
```

### 3. Task Polling
```
Client ──► GET /tasks/{id} ──► Backend returns status
                                    │
         ◄─────────────────────────┘
         │
         ▼
    status == "completed"?
         │
         ├─► Yes: Display image
         │
         └─► No: Poll again (1s interval)
```

---

## Memory Requirements

| Component | VRAM Usage | Notes |
|-----------|-----------|-------|
| SDXL UNet | ~6GB | float16 |
| VAE | ~1GB | float16 |
| IP-Adapter FaceID | ~0.5GB | |
| ControlNet Depth | ~2GB | Optional |
| MiDaS Depth | ~0.5GB | CPU/lazy loaded |
| **Total (without ControlNet)** | ~8GB | |
| **Total (with ControlNet)** | ~10GB | |

### Platform Compatibility

| Platform | dtype | Notes |
|----------|-------|-------|
| CUDA | float16 | 권장 |
| MPS (Mac) | float16 | M1/M2/M3 지원 |
| CPU | float32 | 매우 느림, 비권장 |

---

## Troubleshooting

### MPS dtype mismatch
```
Error: 'mps.add' op requires the same element type
```
**해결**: ControlNet dtype을 float16으로 통일 (v5에서 수정됨)

### FaceID가 너무 약함
**원인**: `ip_adapter_scale`이 하드코딩되어 있었음
**해결**: 파라미터로 전달되도록 수정 (v5에서 수정됨)

### 얼굴이 검출되지 않음
```
Error: No face detected in the input image
```
**해결**: 얼굴이 명확히 보이는 이미지 사용, 최소 112x112 해상도 권장

### 스타일이 반영되지 않음
**원인**: `style_strength`가 너무 낮음
**해결**: 0.3-0.5 범위로 조정

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0 | 2025-05-28 | 논문 원본 구현 |
| v1 | 2025-12-05 | Mac MPS 지원, Web UI |
| v2 | 2025-12-08 | RealVisXL, 비동기 Task |
| v3 | 2025-12-08 | Batch-Wise DCG |
| v4 | 2025-12-11 | CLIP Blending (Identity Loss 해결) |
| v5 | 2025-12-11 | ControlNet 통합, 구조 보존 |

---

## References

- [FastFace Paper](https://arxiv.org/) - Decoupled Guidance 논문
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) - 이미지 프롬프트 어댑터
- [ControlNet](https://github.com/lllyasviel/ControlNet) - 구조 제어
- [InsightFace](https://github.com/deepinsight/insightface) - 얼굴 분석

---

## Related Issues

- [Issue #1](https://github.com/danlee-dev/prometheus-fastface-dev/issues/1) - Identity Loss 문제 (해결됨, v4)
- [Issue #2](https://github.com/danlee-dev/prometheus-fastface-dev/issues/2) - 인물 없는 스타일 이미지 문제 (진행중)
