# NInfer SM86 v0.7.0-sm86

**Published:** 2026-09-26 · **Platform:** Windows x64 · **CUDA packages:** CUDA 12 and CUDA 13

## Problem this release addresses

The upstream project targets Blackwell (`sm_120a`), so Ampere GPU owners need a separate SM86
build/runtime path to use NInfer rather than the Blackwell-targeted build. This fork provides that
compatibility route for supported Qwen `.ninfer` artifacts; this release packages it for Windows
CUDA 12/13 and documents Linux source/Docker builds. It is aimed at running inference on existing
single-GPU Ampere hardware with bounded memory and concurrency—not at adding multi-GPU execution or
an unrestricted datacenter scheduler.


## Downloads

- [CUDA 12 Windows archive](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda12.zip)
- [CUDA 13 Windows archive](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda13.zip)
- [All release assets](https://github.com/doha-230/ninfer-sm86/releases/tag/v0.7.0-sm86)

Choose the archive compatible with the installed NVIDIA driver/toolkit environment. The archive contains the Windows applications and runtime dependencies, not model weights. See the [Windows guide](docs/rtx-3090-windows.md) for model download and launch instructions.

## Product scope

This release line targets SM86; RTX 3090/3090 Ti are the primary measured GPUs. Other SM86 cards can have materially different VRAM capacity and are not implied to share RTX 3090 capacity results.

Registered model and weight identities in the source line are:

- Qwen3.6-27B: `groupwise-int`, `nvfp4`
- Qwen3.8-27B: `groupwise-int`, `nvfp4`
- Qwen3.6-35B-A3B: `groupwise-int` (DFlash is text-only)

The runtime includes paged BF16/INT8 KV, bounded request cohorts, CUDA Graph decode, prefix reuse, speculative execution when supported by the artifact, multimodal inputs on supported routes, and OpenAI Chat Completions/Responses plus Anthropic Messages APIs. NInfer returns tool calls; it does not execute tools.

## Build requirements and platform status

- Source build: CUDA Toolkit 12.8 or newer and CMake 3.28 or newer.
- Linux: Docker or native source build; this release does not include a prebuilt Linux archive.
- Windows: use the CUDA 12 or CUDA 13 archive above, or build with Visual Studio 2022 and vcpkg.
- The release CI built the two Windows CUDA variants. That is compile/package evidence, not a claim that every model/profile has been requalified on an RTX 3090.

## Known SM86 limitations

- Blackwell-only NVFP4/W4A4 and FP8 A8 Tensor Core paths are unavailable. FP8/NVFP4 weights use supported A16 dequantization routes.
- FP8 E4M3 KV and RotorQuant `rk8v4` are not supported by the current KV-cache implementation. Use BF16 or INT8 KV.
- No multi-GPU execution, CPU weight offload, or unrestricted preemptive continuous batching.
- Memory fit and safe concurrency depend on artifact, options, and other GPU allocations; start with the model-specific guidance.

## Further reading

- [Project overview (English)](README.md) · [한국어](README.ko.md)
- [Linux guide](docs/rtx-3090-linux.md) · [Windows guide](docs/rtx-3090-windows.md)
- [CLI](docs/cli.md) · [Serving APIs](docs/serving.md) · [Performance methodology](docs/performance.md)
- [CUDA 12 release asset](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda12.zip) · [CUDA 13 release asset](https://github.com/doha-230/ninfer-sm86/releases/download/v0.7.0-sm86/ninfer-sm86-cuda13.zip)

## Release CI

The `v0.7.0-sm86` Windows workflow completed its CUDA 12 and CUDA 13 build/package jobs. The current repository CI is being expanded separately to run host Python tests and compile the full SM86 source tree; real CUDA kernel tests require an available NVIDIA GPU runner. See [GitHub Actions](https://github.com/doha-230/ninfer-sm86/actions) for current run status.
