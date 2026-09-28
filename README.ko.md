# NInfer SM86

[English](README.md) | **한국어**

## 이 포크가 해결하는 문제

upstream NInfer는 Blackwell(`sm_120a`)을 대상으로 합니다. 따라서 Ampere GPU에서 실행하려면 호환되는 `sm_86` 빌드와 런타임이 필요합니다. **이 포크는 등록된 Qwen `.ninfer` 아티팩트를 Ampere에서 실행할 수 있는 경로를 제공합니다.** CUDA 12/13용 Windows 패키지와 Linux 소스/Docker 빌드를 지원합니다.

목표는 기존 GPU에서 제한된 메모리와 요청 수를 관리하며 단일 GPU 추론을 하는 것입니다. 멀티 GPU나 데이터센터 규모의 스케줄러를 제공하려는 프로젝트는 아닙니다. RTX 3090/3090 Ti를 주 기준 장치로 사용하며, 메모리가 더 적은 다른 `sm_86` GPU에서는 컨텍스트나 동시 요청 수를 낮춰야 할 수 있습니다.

## 지원 모델

| 모델 | 등록된 가중치 ID | 참고 |
|---|---|---|
| Qwen3.6-27B | `groupwise-int`, `nvfp4` | 텍스트·비전 |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` | 텍스트·비전 |
| Qwen3.6-35B-A3B | `groupwise-int` | 텍스트·비전; DFlash는 텍스트 전용 |

아티팩트 ID, 다운로드, 호환성은 모델 카드에서 확인하세요.

- [Qwen3.6-27B — groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md)
- [Qwen3.8-27B — groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md)
- [Qwen3.6-35B-A3B — groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md)

## 제공 기능

- 등록 모델과 `.ninfer` 아티팩트를 위한 공통 `sm_86` 런타임.
- 아티팩트가 지원하는 텍스트·비전·prefix 재사용·MTP·CLI/HTTP 추론 경로.
- Paged BF16 또는 INT8 group-64 KV cache, CUDA Graph decode, ReplaySSM 상태 처리.
- 시작 시 정하는 1–8개 요청의 제한된 동시 처리와 대기 요청 admission.
- OpenAI Chat Completions/Responses 및 Anthropic Messages API. Tool call은 반환하지만 도구 실행은 하지 않습니다.

## 다운로드 및 실행

1. **Windows:** [최신 릴리스](https://github.com/doha-230/ninfer-sm86/releases/latest)에서 CUDA 12 또는 CUDA 13용 x64 아카이브를 받으세요. 모델 가중치는 별도로 받아야 합니다.
2. 위 모델 카드에서 호환되는 `.ninfer` 모델을 다운로드하세요.
3. 포함된 실행 스크립트를 사용하거나 [Windows 안내서](docs/rtx-3090-windows.md)를 따르세요.
4. **Linux:** Docker 또는 [Linux 안내서](docs/rtx-3090-linux.md)에 따른 소스 빌드를 사용하세요. 사전 빌드 Linux 아카이브는 배포하지 않습니다.

소스 빌드에는 CUDA Toolkit 12.8 이상과 CMake 3.28 이상이 필요합니다. CLI 옵션과 요청 형식은 [CLI 안내](docs/cli.md)와 [HTTP serving 안내](docs/serving.md)를 참고하세요.

## 제한사항과 검증

- 프로세스당 CUDA GPU 한 장과 상주 모델 하나를 사용합니다. 멀티 GPU와 CPU weight offload는 지원하지 않습니다.
- 동시성은 시작 시 제한하며, 선점형 대규모 continuous batching은 지원하지 않습니다.
- Blackwell 전용 NVFP4/W4A4 및 FP8 A8 Tensor Core 경로는 `sm_86`에서 사용할 수 없습니다. FP8/NVFP4 가중치는 지원되는 A16 dequantize 경로를 사용합니다.
- KV cache는 BF16 또는 INT8을 사용하세요. FP8 E4M3 KV와 RotorQuant `rk8v4`는 지원하지 않습니다.
- 실행 가능 용량은 아티팩트, 컨텍스트, 옵션, 다른 GPU 작업에 따라 다릅니다. 과거 RTX 3090 측정값은 현재 소스나 모든 릴리스 바이너리의 검증 결과가 아닙니다.
- Hosted CI는 Python 테스트와 CUDA `sm_86` 컴파일을 검사합니다. 실제 GPU CTest에는 self-hosted NVIDIA runner가 필요하지만 현재 등록된 runner가 없습니다. 따라서 컴파일 성공만으로 GPU 커널의 정확성이 입증되지는 않습니다.

## 링크

- [최신 릴리스](https://github.com/doha-230/ninfer-sm86/releases/latest)
- [릴리스 노트 — English](RELEASE_NOTES_0.7.0-sm86.md) · [한국어](RELEASE_NOTES_0.7.0-sm86.ko.md)
- [문서 목차](docs/README.md) · [기여 정책](PR_POLICY.md)
- [Upstream NInfer](https://github.com/Neroued/ninfer)
- Apache License 2.0 — [LICENSE](LICENSE)
