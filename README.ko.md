# NInfer SM86

[English](README.md) | **한국어**

NInfer SM86은 **Ampere GPU 호환성 공백**을 해결합니다. upstream NInfer는 Blackwell(`sm_120a`)을
대상으로 하며, 이 저장소가 유지하는 Ampere `sm_86` 빌드·런타임 경로는 제공하지 않습니다.
이 포크는 그 경로를 구현해 RTX 30 시리즈 등 Ampere GPU에서 등록된 Qwen `.ninfer` 모델을
단일 GPU로 실행할 수 있게 합니다. CUDA 12/13 Windows 패키지와 Linux 소스 빌드를 제공합니다.
목표는 기존 GPU에서 제한된 메모리와 동시 요청을 관리하며 추론하는 것이지, 멀티 GPU나
데이터센터급 선점형 배칭이 아닙니다. 성능 측정은 주로 RTX 3090/3090 Ti에서 수행했으며,
VRAM이 더 적은 카드에서 같은 결과를 기대할 수는 없습니다.

## 무엇을 지원하나요?

엔진은 등록된 `.ninfer` 아티팩트를 로드하고 하나의 런타임 경로로 추론합니다.

| 모델 | 지원 가중치 ID | 참고 |
|---|---|---|
| Qwen3.6-27B | `groupwise-int`, `nvfp4` | 텍스트·비전 |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` | 텍스트·비전 |
| Qwen3.6-35B-A3B | `groupwise-int` | 텍스트·비전; DFlash는 텍스트 전용 |

모델 카드: [3.6-27B groupwise](model-cards/Qwen3.6-27B-NInfer/README.md) · [3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) · [3.8-27B groupwise](model-cards/Qwen3.8-27B-NInfer/README.md) · [3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) · [3.6-35B-A3B](model-cards/Qwen3.6-35B-A3B-NInfer/README.md)

지원 기능은 paged BF16 또는 INT8 group-64 KV cache, 최대 1–8개 요청의 제한된 동시 실행, CUDA Graph decode, prefix 재사용, 아티팩트에 포함된 speculative decoding(MTP), ReplaySSM 상태 처리, 이미지/비디오 입력, OpenAI Chat Completions/Responses 및 Anthropic Messages 호환 API입니다. Tool call은 클라이언트에 반환하지만 NInfer가 도구를 직접 실행하지는 않습니다.

## 지원 범위와 제한

- 단일 프로세스·단일 GPU·단일 상주 모델 기준입니다. 멀티 GPU, CPU weight offload, 무제한 선점형 continuous batching은 지원하지 않습니다.
- Blackwell 전용 NVFP4/W4A4 및 FP8 A8 Tensor Core 실행 경로는 `sm_86`에서 사용할 수 없습니다. FP8/NVFP4 가중치는 지원되는 A16 dequantize 경로로 처리합니다.
- KV cache는 BF16 또는 INT8을 사용하세요. FP8 E4M3 KV와 RotorQuant `rk8v4`는 현재 사용할 수 없습니다.
- VRAM과 안전한 동시성 한도는 모델 아티팩트와 시작 옵션에 따라 달라집니다. 엔진은 용량을 초과하는 설정을 거부할 수 있습니다.
- README의 RTX 3090 처리량·컨텍스트 숫자는 특정 과거 빌드, 아티팩트, 실행 설정의 측정값입니다. 최신 소스 또는 모든 릴리스 바이너리를 GPU에서 재인증했다는 뜻이 아닙니다. 재현 조건은 [영문 README](README.md)와 [성능 문서](docs/performance.md)를 확인하세요.

## 다운로드 및 빌드

현재 최신 공개 릴리스는 [v0.7.0-sm86](https://github.com/doha-230/ninfer-sm86/releases/tag/v0.7.0-sm86)입니다. CUDA 12와 CUDA 13용 Windows x64 아카이브를 제공합니다. 모델 파일은 릴리스 아카이브에 포함되지 않으므로 별도로 받아야 합니다.

Linux는 Docker 또는 소스 빌드를 사용합니다. 이 저장소는 사전 빌드 Linux 바이너리를 배포하지 않습니다. 소스 빌드에는 CUDA Toolkit 12.8 이상과 CMake 3.28 이상이 필요합니다.

- [Linux 빌드·실행 안내](docs/rtx-3090-linux.md)
- [Windows 설치·실행 안내](docs/rtx-3090-windows.md)
- [CLI 옵션](docs/cli.md)
- [HTTP API 안내](docs/serving.md)
- [영문 README 및 과거 성능표](README.md)

## 검증 상태

GitHub Actions는 Python 테스트와 CUDA 12.8 `sm_86` 전체 빌드를 실행합니다. 이 빌드는 CUDA GPU가 없어도 컴파일을 검증할 수 있지만, 커널의 실제 GPU 수치 검증을 대신하지 않습니다. 전체 GPU CTest job은 `sm_86` GPU가 연결된 self-hosted runner에서만 실행할 수 있습니다. 결과는 [Actions](https://github.com/doha-230/ninfer-sm86/actions)에서 확인하세요.

## 프로젝트

이 저장소는 [Neroued/ninfer](https://github.com/Neroued/ninfer)의 커뮤니티 포크이며 Ampere `sm_86`에 맞춘 호환 코드와 실행 경로를 제공합니다. 유지보수와 지원은 최선의 노력으로 제공됩니다.

- [v0.7.0-sm86 릴리스 노트 (English)](RELEASE_NOTES_0.7.0-sm86.md) · [한국어](RELEASE_NOTES_0.7.0-sm86.ko.md)
- [기여 정책](PR_POLICY.md)
- [Apache License 2.0](LICENSE)
