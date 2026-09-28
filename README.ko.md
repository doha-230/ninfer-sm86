# NInfer SM86

[English](README.md) · **한국어**

Ampere GPU(Compute Capability **8.6**)에서 지원 Qwen `.ninfer` 모델을 실행하세요. 이 저장소는 Blackwell 중심 NInfer에서 파생된 Don-Chad의 SM86 포크를 기반으로 합니다. SM86 지원을 이 포크가 처음 만든 것은 아닙니다.

> **시작 안내:** RTX 3090/3090 Ti 및 RTX A6000 사용자는 Windows 패키지와 모델 아티팩트를 각각 다운로드해 로컬 CLI 또는 HTTP 서버를 실행할 수 있습니다. Linux에서는 Docker나 소스 빌드를 사용합니다. 모델 가중치는 패키지에 포함되지 않습니다.

## Don-Chad 원본과 비교해 달라진 점

[Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090)은 이미 RTX 3090용 `sm_86` Qwen 추론, Windows·Linux 바이너리, 주요 서버 기능을 제공했습니다. 이 포크가 해당 기능을 처음 구현했다고 주장하지 않습니다.[1][2] 이 저장소에서 추가로 해결한 실용적 과제는 다음과 같습니다.

- **CUDA 12와 CUDA 13용 Windows x64 릴리스 아카이브를 분리**해 드라이버 환경에 맞는 패키지를 선택할 수 있게 했습니다.[4]
- 설정된 요청 본문 제한보다 먼저 HTTP 413을 내던 form-urlencoded의 별도 8 KiB 제한을 제거했습니다. 설정된 일반 용량 제한은 유지합니다. OpenAI 클라이언트는 단일 탑재 모델에 대해 비어 있지 않은 별칭을 보낼 수 있고, 응답에는 서버의 공개 모델 ID를 사용합니다.
- `ninfer-serve --chat-template FILE`로 로컬 Jinja 템플릿을 아티팩트 수정 없이 사용합니다. 비전에는 `--vision`과 이미지·비디오 자리표시자를 원래 입력 순서대로 출력하는 템플릿이 필요하며, 정확한 위치를 알 수 없는 캐시 마커는 거절합니다.
- GitHub 호스팅 러너에서 **Python 테스트와 CUDA 12.8 `sm_86` 전체 빌드**를 검사합니다. GPU CTest는 사용할 수 있는 self-hosted GPU runner가 있을 때만 실행됩니다. 이는 GPU 실행 검증과 다릅니다.
- **RTX A6000(Ampere, `sm_86`) 동작을 사용자로부터 확인**했습니다. A6000 전용 커널, 성능 수치 또는 모든 모델·설정 조합의 검증을 뜻하지는 않습니다.

Don-Chad의 v0.6.1은 Linux 바이너리도 배포했지만, 이 저장소의 현재 릴리스에는 Windows 아카이브만 있습니다. Linux는 소스 또는 Docker로 빌드합니다.[2][4]

## 내 GPU에서 실행할 수 있나요?

빌드 대상은 Ampere `sm_86`이지 모든 RTX GPU가 아닙니다. NVIDIA의 Compute Capability 목록에 따라 아래 장치를 구분했습니다. 아키텍처가 맞는다는 것과 대형 Qwen 아티팩트가 VRAM에 들어간다는 것은 다릅니다.[3]

| GPU | 이 저장소에서의 상태 | 주의할 점 |
|---|---|---|
| GeForce RTX 3090 / 3090 Ti | Don-Chad에서 이어진 RTX 3090 SM86 경로; 주요 기준 제품군 | 24 GB급. 실제 여유 메모리는 아티팩트와 컨텍스트에 따라 다릅니다. |
| NVIDIA RTX A6000 (Ampere) | **사용자가 A6000에서 동작 확인**. 설정과 전체 테스트 항목은 기록되지 않았습니다. | 48 GB급. 성능이나 최대 컨텍스트 검증으로 확대 해석하지 마세요. |
| GeForce RTX 3080 / 3080 Ti, 3070 / 3070 Ti, 3060 / 3060 Ti; RTX A5000 / A4000 / A2000 | `sm_86` 아키텍처 후보이며 **이 저장소에서 검증하지 않음** | VRAM이 제각각입니다. 나열된 27B 아티팩트 자체가 런타임 메모리 외에 약 16~17 GiB를 차지하므로 많은 구성에서 적재가 불가능합니다. |

가중치를 받기 전에 GPU의 Compute Capability와 VRAM을 확인하세요. `sm_80`(예: A100)과 `sm_89`(예: RTX 4090)는 `sm_86` 릴리스 아카이브의 대상이 아닙니다. 이 표는 NVIDIA의 모든 8.6 장치를 나열하지 않습니다.[3] 작은 컨텍스트와 단일 요청에서 시작하고, 모델 적재 성공과 전체 경로 검증을 구분해야 합니다.

| 모델 | 아티팩트 프로필 | 참고 |
|---|---|---|
| Qwen3.6-27B | `groupwise-int`, `nvfp4` | 텍스트·비전 |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` | 텍스트·비전 |
| Qwen3.6-35B-A3B | `groupwise-int` | 텍스트·비전; DFlash는 텍스트 전용 |

아티팩트는 [모델 카드](#모델-아티팩트)에서 선택하세요. 연결된 카드는 아티팩트의 식별 정보와 출처를 설명하며, 일부는 upstream Blackwell 실행 환경을 기준으로 작성됐습니다. Ampere에서는 이 문서의 SM86 런타임과 제약사항을 따르세요. `.ninfer`는 NInfer 전용 형식입니다. Transformers 체크포인트, Safetensors 디렉터리, GGUF 파일을 그대로 사용할 수 없습니다.

## 빠른 시작

### Windows x64

1. [최신 릴리스](https://github.com/doha-230/ninfer-sm86/releases/latest)에서 NVIDIA 드라이버에 맞는 CUDA 12 또는 CUDA 13용 Windows x64 아카이브를 받아 압축을 풉니다.
2. [Qwen3.8-27B groupwise-int](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/blob/main/qwen3_8_27b.ninfer) 아티팩트(약 17 GiB)를 다운로드합니다. 압축을 푼 실행 파일 옆의 `models\qwen3_8_27b.ninfer`로 저장하세요. 체크섬은 [모델 카드](model-cards/Qwen3.8-27B-NInfer/README.md)에 있습니다.
3. 해당 디렉터리에서 PowerShell을 열고 짧은 텍스트 생성을 실행합니다.

```powershell
.\ninfer.exe .\models\qwen3_8_27b.ninfer `
  --prompt "Explain prefill and decode in two sentences." `
  --max-context 8192 --max-new 128 `
  --kv-dtype int8
```

같은 아티팩트로 로컬 HTTP 서버를 시작하려면:

```powershell
.\ninfer-serve.exe .\models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 `
  --max-concurrency 1 --kv-dtype int8
```

요청 예제는 [HTTP serving 안내](docs/serving.md)를 참고하세요. 아카이브에는 Windows 실행 파일과 런타임 의존성이 포함되며 **모델 가중치는 포함되지 않습니다.** Microsoft Visual C++ 2022 런타임이 없다면 별도로 설치해야 합니다.

### Linux

사전 빌드 Linux 릴리스 아카이브는 없습니다. Docker 또는 소스에서 빌드하고 모델 아티팩트를 별도로 다운로드하세요. [Linux 안내서](docs/rtx-3090-linux.md)에 필수 구성요소, 빌드 명령, Docker 실행 예제와 짧은 생성 확인 절차가 있습니다. 안내서의 과거 릴리스 관련 내용은 현재 Windows 릴리스의 설명이 아닙니다.

## 모델 아티팩트

아래 모델 카드는 등록된 아티팩트, 다운로드 방법, 파일명과 체크섬을 안내합니다. upstream 모델 카드의 런타임 요구사항은 이 포크의 SM86 호환 범위를 설명하지 않습니다.

- Qwen3.6-27B: [groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md)
- Qwen3.8-27B: [groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md)
- Qwen3.6-35B-A3B: [groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md)

## 런타임 기능

- 텍스트 생성, 그리고 아티팩트가 지원하는 경로에서 이미지·비디오 입력.
- 로컬 CLI 및 OpenAI Chat Completions/Responses, Anthropic Messages 호환 HTTP API.
- 선택형 MTP speculative decoding, prefix 재사용, CUDA Graph decode, 제한된 동시 요청 처리.
- BF16 또는 INT8 paged KV cache. Tool call은 클라이언트에 반환하지만 NInfer가 도구를 실행하지는 않습니다.

기능은 선택한 아티팩트와 시작 옵션에 따라 달라집니다. 프로세스 하나는 GPU 하나에 모델 하나를 올리며, 동시 요청 수는 시작 시 제한합니다. 멀티 GPU 추론이나 선점형 대규모 continuous batching을 제공하는 서버는 아닙니다. 자세한 동작과 옵션은 [CLI 안내](docs/cli.md)와 [HTTP serving 안내](docs/serving.md)를 참고하세요.

## SM86 제약 및 검증 범위

- Blackwell 전용 NVFP4/W4A4 및 FP8 A8 Tensor Core 실행 경로는 SM86에서 사용할 수 없습니다. NVFP4/FP8 가중치 아티팩트는 지원되는 A16 dequantize 경로를 사용합니다. 가중치 프로필이 Blackwell 커널 지원을 뜻하지는 않습니다.
- KV cache는 BF16과 INT8을 지원합니다. FP8 E4M3 KV와 RotorQuant `rk8v4`는 지원하지 않습니다.
- 사용 가능한 VRAM, 컨텍스트 길이와 안전한 동시성은 GPU 및 작업에 따라 다릅니다. 과거 RTX 3090 측정값은 모든 SM86 GPU, 현재 소스 버전 또는 모든 릴리스 바이너리에서의 결과를 보장하지 않습니다. [성능 측정 방법](docs/performance.md)을 참고하세요.
- CI는 Python 테스트와 CUDA SM86 컴파일을 수행합니다. 실제 CUDA 테스트에는 NVIDIA GPU runner가 필요하지만 현재 등록되어 있지 않습니다. 따라서 CI 컴파일만으로 GPU 런타임 정확성이 검증된 것은 아닙니다.

## 프로젝트 링크

- [최신 릴리스](https://github.com/doha-230/ninfer-sm86/releases/latest) · [English 릴리스 노트](RELEASE_NOTES_0.7.0-sm86.md) · [한국어 릴리스 노트](RELEASE_NOTES_0.7.0-sm86.ko.md)
- [문서 목차](docs/README.md) · [기여 정책](PR_POLICY.md) · [Upstream NInfer](https://github.com/Neroued/ninfer)
- Apache License 2.0 · [LICENSE](LICENSE)

## Sources

[1] https://github.com/Don-Chad/ninfer-3090 — Don-Chad NInfer-3090
[2] https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.6.1-rtx3090 — Don-Chad v0.6.1 release
[3] https://developer.nvidia.com/cuda-gpus — NVIDIA CUDA GPUs
[4] https://github.com/doha-230/ninfer-sm86/releases/tag/v0.7.0-sm86 — NInfer SM86 v0.7.0 release
