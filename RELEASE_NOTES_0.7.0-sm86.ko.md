# NInfer SM86 v0.7.0-sm86

**배포일:** 2026-09-26 · **플랫폼:** Windows x64 · **CUDA 패키지:** CUDA 12, CUDA 13

## 이 릴리스가 해결하는 문제

upstream 프로젝트는 Blackwell(`sm_120a`)을 대상으로 하므로 Ampere GPU에서 NInfer를 사용하려면
별도의 `sm_86` 빌드·런타임 경로가 필요합니다. 이 포크는 지원 Qwen `.ninfer` 아티팩트용
호환 경로를 제공하며, 이 릴리스는 Windows CUDA 12/13 패키지와 Linux 소스/Docker 빌드 안내를
제공합니다. 기존 Ampere 단일 GPU에서 메모리와 동시 요청을 제한해 추론하는 것이 목적이며,
멀티 GPU나 제한 없는 데이터센터 스케줄러를 추가하는 릴리스는 아닙니다.

이번 릴리스는 Windows용 아카이브 두 가지를 제공합니다. Linux에서는 Docker 또는 소스 빌드를 사용하세요. 모델 아티팩트는 별도로 다운로드해야 합니다.

## 다운로드

- [CUDA 12 Windows 아카이브](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda12.zip)
- [CUDA 13 Windows 아카이브](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda13.zip)
- [전체 릴리스 파일](https://github.com/doha-230/ninfer-sm86/releases/tag/v0.7.0-sm86)

설치된 NVIDIA 드라이버와 CUDA 환경에 맞는 아카이브를 선택하세요. 아카이브에는 Windows 실행 파일과 런타임 DLL이 포함되지만 모델 가중치는 포함되지 않습니다. 모델 다운로드와 실행 방법은 [Windows 안내서](docs/rtx-3090-windows.md)를 참고하세요.

## 제품 범위

이 릴리스 계열은 SM86을 대상으로 하며 RTX 3090/3090 Ti를 주요 측정 GPU로 사용합니다. 다른 SM86 GPU는 VRAM이 크게 다를 수 있으므로 RTX 3090의 메모리·컨텍스트 측정값을 그대로 적용할 수 없습니다.

소스에 등록된 모델/가중치 ID는 다음과 같습니다.

- Qwen3.6-27B: `groupwise-int`, `nvfp4`
- Qwen3.8-27B: `groupwise-int`, `nvfp4`
- Qwen3.6-35B-A3B: `groupwise-int` (DFlash는 텍스트 전용)

런타임은 paged BF16/INT8 KV, 제한된 요청 동시 처리, CUDA Graph decode, prefix 재사용, 아티팩트에서 지원하는 speculative 실행, 지원 경로의 멀티모달 입력, OpenAI Chat Completions/Responses 및 Anthropic Messages API를 제공합니다. Tool call은 반환하지만 실제 도구 실행은 하지 않습니다.

## 빌드 및 플랫폼

- 소스 빌드: CUDA Toolkit 12.8 이상, CMake 3.28 이상.
- Linux: Docker 또는 네이티브 소스 빌드. 이번 릴리스에는 사전 빌드 Linux 아카이브가 없습니다.
- Windows: 위의 CUDA 12 또는 CUDA 13 아카이브를 사용하거나 Visual Studio 2022와 vcpkg로 빌드합니다.
- 릴리스 CI에서 Windows CUDA 12/13 빌드가 완료됐습니다. 이는 컴파일·패키징 근거이며, 모든 모델 및 실행 프로필을 RTX 3090에서 재검증했다는 의미는 아닙니다.

## SM86 제한사항

- Blackwell 전용 NVFP4/W4A4 및 FP8 A8 Tensor Core 경로는 사용할 수 없습니다. FP8/NVFP4 가중치는 지원되는 A16 dequantize 경로를 사용합니다.
- 현재 KV-cache 구현은 FP8 E4M3 KV와 RotorQuant `rk8v4`를 지원하지 않습니다. BF16 또는 INT8 KV를 사용하세요.
- 멀티 GPU, CPU weight offload, 무제한 선점형 continuous batching은 지원하지 않습니다.
- 메모리 적합성과 안전한 동시성은 아티팩트, 실행 옵션, 다른 GPU 작업에 따라 달라집니다. 모델별 안내에서 시작 설정을 확인하세요.

## 추가 문서

- [프로젝트 안내 (English)](README.md) · [한국어](README.ko.md)
- [Linux 안내서](docs/rtx-3090-linux.md) · [Windows 안내서](docs/rtx-3090-windows.md)
- [CLI](docs/cli.md) · [Serving API](docs/serving.md) · [성능 측정 방법](docs/performance.md)
- [CUDA 12 릴리스 파일](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda12.zip) · [CUDA 13 릴리스 파일](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda13.zip)

## 릴리스 CI

`v0.7.0-sm86` Windows workflow에서 CUDA 12 및 CUDA 13 빌드·패키징 작업이 완료됐습니다. 현재 저장소 CI는 호스트 Python 테스트와 전체 SM86 소스 컴파일 검증으로 확장 중이며, CUDA 커널의 실제 실행 테스트에는 NVIDIA GPU runner가 필요합니다. 최신 결과는 [GitHub Actions](https://github.com/doha-230/ninfer-sm86/actions)에서 확인하세요.
