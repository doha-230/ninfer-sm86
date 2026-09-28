# NInfer SM86

[English](README.md) · **한국어**

Ampere GPU(Compute Capability **8.6**)에서 지원 Qwen `.ninfer` 모델을 실행하세요. 이 포크는 Blackwell 중심의 upstream 프로젝트가 대상으로 삼지 않는 `sm_86` 빌드·런타임 경로를 제공합니다.

> **시작 안내:** RTX 3090/3090 Ti 사용자는 Windows 패키지와 모델 아티팩트를 각각 다운로드해 로컬 CLI 또는 HTTP 서버를 실행할 수 있습니다. Linux에서는 Docker나 소스 빌드를 사용합니다. 모델 가중치는 패키지에 포함되지 않습니다.

## 내 GPU에서 실행할 수 있나요?

소프트웨어 대상은 Ampere `sm_86`이며 RTX 3090과 RTX 3090 Ti를 주요 기준 장치로 삼습니다. 다른 `sm_86` GPU는 VRAM이 훨씬 적을 수 있습니다. 모델 적재 여부와 사용 가능한 컨텍스트 길이는 GPU, 아티팩트, 실행 옵션, 다른 GPU 작업에 따라 달라지므로 작은 컨텍스트와 단일 요청부터 시작하세요.

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
