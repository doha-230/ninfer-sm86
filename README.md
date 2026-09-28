# NInfer SM86

[English](README.md) · [한국어](README.ko.md)

Run supported Qwen `.ninfer` models on Ampere GPUs with compute capability **8.6**. This repository builds on Don-Chad's SM86 fork of the Blackwell-focused NInfer project; SM86 support did **not** originate here.

> **Start here:** RTX 3090 / 3090 Ti and RTX A6000 users can download a Windows package and a model artifact, then run a local CLI or HTTP server. Linux users can build with Docker or from source. Model weights are separate downloads.

## What this fork changes from Don-Chad

[Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090) already brought Qwen inference to `sm_86`, including RTX 3090 builds, Windows and Linux binaries, and the main serving features. This fork does not claim those as new inventions.[1][2] Its additional practical work is:

- Separate **CUDA 12 and CUDA 13 Windows x64 release archives**, so users can select a package for their driver environment.[4]
- GitHub-hosted **Python tests and a complete CUDA 12.8 `sm_86` compile**; GPU CTest remains conditional on an available self-hosted GPU runner. This is build/host-test coverage, not GPU runtime qualification.
- **RTX A6000 (Ampere, `sm_86`) operation confirmed by a user**. This is not a separate A6000 kernel path, a benchmark, or proof that every model and profile works on that device.

Don-Chad's v0.6.1 release included a Linux binary; the current release here supplies Windows archives only. Linux users build from source or Docker.[2][4]

## Is my GPU supported?

The build target is Ampere `sm_86`, not all RTX cards. NVIDIA lists the devices below as compute capability 8.6; architecture eligibility is **not** a claim that these large Qwen artifacts fit in each card's VRAM.[3]

| GPU | Status for this repository | Practical limit |
|---|---|---|
| GeForce RTX 3090 / 3090 Ti | Established RTX 3090 SM86 path inherited from Don-Chad; primary reference class | 24 GB cards; available memory still depends on artifact and context. |
| NVIDIA RTX A6000 (Ampere) | **User-confirmed working** on A6000; configuration and full test matrix not recorded here | 48 GB class; no measured speed or validated maximum context is claimed. |
| GeForce RTX 3080 / 3080 Ti, 3070 / 3070 Ti, 3060 / 3060 Ti; RTX A5000 / A4000 / A2000 | `sm_86` architecture candidates, **not verified here** | VRAM varies; the listed 27B artifacts alone occupy roughly 16–17 GiB before runtime allocations, so many configurations cannot fit. |

Check your GPU's compute capability and VRAM before downloading weights. `sm_80` (for example A100) and `sm_89` (for example RTX 4090) are not covered by the `sm_86` release archives; this table is not an exhaustive list of NVIDIA 8.6 devices.[3] Start with a short context and one request, and distinguish a successful load from full route qualification.

| Model | Artifact profiles | Notes |
|---|---|---|
| Qwen3.6-27B | `groupwise-int`, `nvfp4` | Text and vision |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` | Text and vision |
| Qwen3.6-35B-A3B | `groupwise-int` | Text and vision; DFlash is text-only |

Choose an artifact from its [model card](#model-artifacts). The linked cards document artifact identity and provenance; some were authored for upstream Blackwell builds. For Ampere, use the SM86 runtime and constraints described here. `.ninfer` is NInfer's own format; a Transformers checkpoint, Safetensors directory, or GGUF file cannot be used directly.

## Quick start

### Windows x64

1. Download and extract a Windows x64 archive from the [latest release](https://github.com/doha-230/ninfer-sm86/releases/latest). Choose its CUDA 12 or CUDA 13 variant for your installed NVIDIA driver.
2. Download [Qwen3.8-27B groupwise-int](https://huggingface.co/neroued/Qwen3.8-27B-NInfer/blob/main/qwen3_8_27b.ninfer) (about 17 GiB). Save it as `models\qwen3_8_27b.ninfer` beside the extracted executables. The [model card](model-cards/Qwen3.8-27B-NInfer/README.md) has its checksum.
3. Open PowerShell in that directory and run a short text generation:

```powershell
.\ninfer.exe .\models\qwen3_8_27b.ninfer `
  --prompt "Explain prefill and decode in two sentences." `
  --max-context 8192 --max-new 128 `
  --kv-dtype int8
```

To start a local HTTP server with the same artifact, run:

```powershell
.\ninfer-serve.exe .\models\qwen3_8_27b.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 `
  --max-concurrency 1 --kv-dtype int8
```

See [HTTP serving](docs/serving.md) for request examples. The archive contains Windows applications and runtime dependencies, **not** model weights. Install the Microsoft Visual C++ 2022 runtime if it is not already present.

### Linux

There is no prebuilt Linux release archive. Build with Docker or from source, then download a model artifact separately. The [Linux guide](docs/rtx-3090-linux.md) includes prerequisites, build commands, a Docker run example, and a short generation check. Its older release-specific notes are not a description of the current Windows release.

## Model artifacts

These links identify the registered artifacts and provide download instructions, filenames, and checksums. Runtime requirements shown in an upstream model card do not describe this fork's SM86 compatibility:

- Qwen3.6-27B: [groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md)
- Qwen3.8-27B: [groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md)
- Qwen3.6-35B-A3B: [groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md)

## Runtime capabilities

- Text generation and, on supported artifact routes, image/video input.
- Local CLI plus OpenAI Chat Completions/Responses and Anthropic Messages-compatible HTTP APIs.
- Optional MTP speculative decoding, prefix reuse, CUDA Graph decode, and bounded concurrent requests.
- BF16 or INT8 paged KV cache. Tool calls can be returned to clients; NInfer does not execute tools.

Features depend on the selected artifact and startup options. The server hosts one model on one GPU per process; concurrency is bounded and configured at startup. This is not multi-GPU inference or a preemptive, large-scale continuous-batching service. See [CLI](docs/cli.md) and [HTTP serving](docs/serving.md) for exact behavior and options.

## SM86 constraints and validation

- Blackwell-only NVFP4/W4A4 and FP8 A8 Tensor Core execution is unavailable on SM86. NVFP4/FP8 weight artifacts use supported A16 dequantization routes; the weight profile does not imply Blackwell kernel support.
- KV cache supports BF16 and INT8. FP8 E4M3 KV and RotorQuant `rk8v4` are not supported.
- Available VRAM, context length, and safe concurrency vary by GPU and workload. Historical RTX 3090 measurements are not a guarantee for every SM86 card, current source revision, or release binary; see the [performance methodology](docs/performance.md).
- CI runs Python tests and compiles the CUDA SM86 targets. On-device CUDA tests need an NVIDIA GPU runner; none is currently registered, so CI compilation alone does not qualify GPU runtime correctness.

## Project links

- [Latest release](https://github.com/doha-230/ninfer-sm86/releases/latest) · [English release notes](RELEASE_NOTES_0.7.0-sm86.md) · [한국어 릴리스 노트](RELEASE_NOTES_0.7.0-sm86.ko.md)
- [Documentation index](docs/README.md) · [Contributing policy](PR_POLICY.md) · [Upstream NInfer](https://github.com/Neroued/ninfer)
- Apache License 2.0 · [LICENSE](LICENSE)

## Sources

[1] https://github.com/Don-Chad/ninfer-3090 — Don-Chad NInfer-3090
[2] https://github.com/Don-Chad/ninfer-3090/releases/tag/v0.6.1-rtx3090 — Don-Chad v0.6.1 release
[3] https://developer.nvidia.com/cuda-gpus — NVIDIA CUDA GPUs
[4] https://github.com/doha-230/ninfer-sm86/releases/tag/v0.7.0-sm86 — NInfer SM86 v0.7.0 release
