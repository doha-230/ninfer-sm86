# NInfer SM86

[English](README.md) | [한국어](README.ko.md)

## Why this fork

Upstream NInfer targets Blackwell (`sm_120a`). Ampere users need a compatible `sm_86` build and runtime to run NInfer on GPUs they already own. **This fork provides that path** for registered Qwen `.ninfer` artifacts, including Windows packages for CUDA 12/13 and Linux source/Docker builds.

The goal is practical single-GPU inference with bounded memory and request concurrency—not multi-GPU serving or a datacenter-scale scheduler. RTX 3090/3090 Ti are the primary reference devices; other `sm_86` GPUs may have less memory and need smaller contexts or concurrency.

## Supported models

| Model | Registered weight IDs | Notes |
|---|---|---|
| Qwen3.6-27B | `groupwise-int`, `nvfp4` | Text and vision |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` | Text and vision |
| Qwen3.6-35B-A3B | `groupwise-int` | Text and vision; DFlash is text-only |

Model cards contain artifact identity, download, and compatibility details:

- [Qwen3.6-27B — groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md)
- [Qwen3.8-27B — groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) · [NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md)
- [Qwen3.6-35B-A3B — groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md)

## What it provides

- One `sm_86` runtime for the registered models and `.ninfer` artifacts.
- Text, vision, prefix reuse, MTP, and CLI/HTTP inference routes supported by each artifact.
- Paged BF16 or INT8 group-64 KV cache, CUDA Graph decode, and ReplaySSM state handling.
- Startup-bounded cohorts of 1–8 requests and bounded pending-request admission.
- OpenAI Chat Completions/Responses and Anthropic Messages APIs. Tool calls are returned to the client; tools are not executed by NInfer.

## Download and run

1. **Windows:** download the [latest release](https://github.com/doha-230/ninfer-sm86/releases/latest). It includes CUDA 12 and CUDA 13 x64 archives; model weights are separate.
2. Download a compatible `.ninfer` model using one of the model cards above.
3. Use the included launcher or follow the [Windows guide](docs/rtx-3090-windows.md).
4. **Linux:** build with Docker or from source using the [Linux guide](docs/rtx-3090-linux.md). No prebuilt Linux archive is published.

Source builds require CUDA Toolkit 12.8+ and CMake 3.28+. For exact CLI flags and request formats, see [CLI](docs/cli.md) and [HTTP serving](docs/serving.md).

## Limits and validation

- Single CUDA GPU and one resident model per process; no multi-GPU execution or CPU weight offload.
- Concurrency is bounded and selected at startup; this is not preemptive, large-scale continuous batching.
- Blackwell-only NVFP4/W4A4 and FP8 A8 tensor-core execution is unavailable on `sm_86`. FP8/NVFP4 weights use supported A16 dequantization paths.
- Use BF16 or INT8 KV. FP8 E4M3 KV and RotorQuant `rk8v4` are not supported.
- Capacity depends on the artifact, context, runtime options, and other GPU allocations. RTX 3090 figures from older builds are not a qualification of the current source or every release binary.
- Hosted CI checks Python tests and compiles the CUDA `sm_86` targets. GPU CTest requires a self-hosted NVIDIA runner; none is currently registered, so compile success does not establish on-device kernel correctness.

## Project links

- [Latest release](https://github.com/doha-230/ninfer-sm86/releases/latest)
- [Release notes — English](RELEASE_NOTES_0.7.0-sm86.md) · [한국어](RELEASE_NOTES_0.7.0-sm86.ko.md)
- [Documentation index](docs/README.md) · [Contributing policy](PR_POLICY.md)
- [Upstream NInfer](https://github.com/Neroued/ninfer)
- Apache License 2.0 — see [LICENSE](LICENSE).
