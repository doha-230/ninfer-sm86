# NInfer SM86

**English** | [한국어](README.ko.md)

NInfer SM86 addresses a hardware-compatibility gap: upstream NInfer targets Blackwell (`sm_120a`) and
does not provide the Ampere `sm_86` build/runtime path maintained here. This fork carries that path
to Ampere—including CUDA 12/13 Windows packages and Linux source builds—so owners of RTX 30-series/
Ampere cards can run registered Qwen `.ninfer` models on one GPU. It is designed around bounded
memory and request concurrency, not multi-GPU or datacenter-scale preemptive batching. The RTX 3090/
3090 Ti is the primary measured device; results do not generalize automatically to lower-memory
Ampere cards.

The engine loads registered `.ninfer` artifacts and serves inference through one runtime. Supported
model/weight identities are Qwen3.6-27B (groupwise-int, NVFP4), Qwen3.8-27B (groupwise-int,
NVFP4), and Qwen3.6-35B-A3B (groupwise-int). The 35B-A3B target also supports text-only DFlash.
All targets support text, vision, MTP where present in the artifact, prefix reuse, and the shared
CLI/HTTP engine route.

**Included runtime features:** paged BF16 or INT8 group-64 KV cache, bounded 1–8 request cohorts,
CUDA Graph decode, compatible-prefix reuse, MTP speculative decoding, ReplaySSM state handling,
image/video input on the supported vision route, and OpenAI Chat Completions/Responses plus
Anthropic Messages APIs. Tool calls are returned to clients; NInfer does not execute tools.

**Platform and qualification:** source builds are supported on Linux and Windows; the published
v0.7.0-sm86 release provides Windows x64 archives for CUDA 12 and CUDA 13. CUDA 12.8+ and
CMake 3.28+ are required to build from source. RTX 3090 results in this README are measurements for
that device and the stated artifact/configuration, not guarantees for every Ampere GPU.

This is a community fork of [Neroued/ninfer](https://github.com/Neroued/ninfer), adapted for
SM86. Community contributions are welcome; maintenance and support are best-effort.

> **SM86 limits:** Blackwell-only NVFP4/W4A4 and FP8 A8 tensor-core execution are unavailable.
> FP8/NVFP4 weights use supported A16 dequantizing routes. FP8 E4M3 KV and RotorQuant `rk8v4`
> are not available; use BF16 or INT8 KV. No multi-GPU execution, CPU weight offload, or
> unrestricted preemptive continuous batching.

## Quick links

- [한국어 안내](README.ko.md)
- [Latest Windows release](https://github.com/doha-230/ninfer-sm86/releases/latest)
- [Linux build and run guide](docs/rtx-3090-linux.md)
- [Windows build and run guide](docs/rtx-3090-windows.md)
- [CLI options](docs/cli.md) · [Serving APIs](docs/serving.md) · [Performance methodology](docs/performance.md)
- [Release notes, v0.7.0-sm86 (English)](RELEASE_NOTES_0.7.0-sm86.md) · [한국어](RELEASE_NOTES_0.7.0-sm86.ko.md)

On the measured RTX 3090 configuration, Qwen3.8-27B reaches a measured **171,648-token INT8
context** with the documented 1 GiB safety headroom. Capacity depends on artifact, runtime options,
driver, and other GPU allocations.

> Historical RotorQuant measurements below describe an older build only. `rk8v4` is rejected by
> the current KV-cache implementation; those results are not supported by the current runtime.

## Choose a platform

| Platform | Delivery | Guide |
|---|---|---|
| Linux | Docker image or native source build; no official prebuilt Linux release | [Linux build guide](docs/rtx-3090-linux.md) |
| Windows 11 | v0.7.0-sm86 CUDA 12 / CUDA 13 archives | [Windows guide](docs/rtx-3090-windows.md) |

### Linux

The Dockerfile gives the shortest build path on Bazzite and other Linux distributions:

```bash
docker build --tag ninfer-3090:sm86 .
```

The Linux guide contains the GPU check, native Ubuntu build, model mount, server command, and Bash
launchers. The project does not publish a prebuilt Linux archive or qualified Linux performance
results yet.

### Windows 11

1. Download and unzip the latest [Windows release](https://github.com/doha-230/ninfer-sm86/releases/latest).
2. Double-click `download-qwen38.bat` to download the model. Interrupted downloads resume.
3. Double-click one launcher:

| Launcher | Best for |
|---|---|
| `run-qwen38-c1.bat` | One interactive user, lowest latency, up to 64K context |
| `run-qwen38-c8.bat` | Multiple users or agents, highest aggregate throughput, 8K context |
| `run-qwen38-vision.bat` | Qwen3.8 image understanding, one user, 32K context, MTP3 |
| `run-qwen36-35b-vision.bat` | Image understanding with Qwen3.6-35B-A3B, one user, 32K context |

The API is then available at `http://127.0.0.1:8080/v1`. The Windows archive includes the required
applications and DLLs.

## RTX 3090 performance reference — historical builds

**Read this qualification note first:** the throughput, capacity, and vision figures in the following
sections were measured on earlier SM86 builds and named artifacts/configurations. They are useful as
historical reference points, not as a requalification of the current post-upstream-sync source tree
or of every v0.7.0-sm86 release archive. No GPU-backed CI runner is currently registered.

For the current build, use the release-specific hardware notes and reproduce measurements with the
commands/methodology in [performance documentation](docs/performance.md). Do not infer that a
feature or number is qualified merely because the current source compiles.

### Qwen3.8-27B support and RTX 3090 results

Qwen3.8-27B is validated from one through eight simultaneous users. ReplaySSM cuts the memory cost
of speculative decoding, allowing the faster MTP3 mode to remain enabled at C8. The table below is
the new sustained test: every request generated 1,024 tokens with CUDA Graphs enabled.

The prompts were **29-34 input tokens** and the server's maximum context window was **8,192 tokens
per request**. Each measured sequence therefore reached roughly 1,053-1,058 tokens including its
generated output. This is a long-output/decode benchmark, not an 8K-prompt or long-prefill test.
C1-C4 used an 8,192-token shared KV pool; C8 used 16,384 tokens so all eight requested outputs
could be admitted simultaneously.

| Cohort | Total output | End-to-end throughput | Decode throughput | MTP acceptance | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1,024 tokens | **70.19 tok/s** | **71.00 tok/s** | 61.13% | 149 ms | 19,641 MiB |
| C2 | 2,048 tokens | **89.43 tok/s** | **90.66 tok/s** | 59.66% | 262 ms | 20,022 MiB |
| C4 | 4,096 tokens | **97.89 tok/s** | **100.28 tok/s** | 59.63% | 538 ms | 20,641 MiB |
| C8 | 8,192 tokens | **161.28 tok/s** | **165.33 tok/s** | 56.84% | 1,215 ms | 22,138 MiB |

C1 is the responsive choice for a single user. C8 delivers **2.3x the total throughput** when
several requests are active. The C8 long-output test uses a 16K shared KV pool so all eight
1,024-token responses can be admitted together.

### Prompt-processing speed

Prompt processing was tested separately with **4,362 fresh input tokens per request**, an 8,192-token
per-request context window, 512-token prefill chunks, INT8 KV, ReplaySSM/MTP3, CUDA Graphs, and
prefix reuse disabled. Each request generated only 16 tokens so the run measures prefill rather
than long decode.

| Cohort | Total fresh input | Aggregate prefill | Active-prefill speed | Mean TTFT | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| C1 | 4,362 tokens | **861.51 tok/s** | 893.98 tok/s | 4,893 ms | 19,114 MiB |
| C2 | 8,724 tokens | **853.86 tok/s** | 883.95 tok/s | 7,478 ms | 19,697 MiB |
| C4 | 17,448 tokens | **847.26 tok/s** | 874.49 tok/s | 12,692 ms | 20,894 MiB |
| C8 | 34,896 tokens | **844.10 tok/s** | 870.94 tok/s | 23,028 ms | 23,207 MiB |

`Aggregate prefill` is total fresh input tokens divided by the complete request-wave time, so it is
the user-facing throughput number. NInfer currently processes one long prefill at a time; cohort
batching accelerates decode, but does not multiply prompt ingestion. Consequently C1-C8 remain near
844-862 input tok/s while queued requests increase mean TTFT. `Active-prefill speed` excludes queue
waiting and measures only the server's recorded prefill phase.

### RotorQuant KV (`rk8v4`) — currently unavailable

`rk8v4` was an experimental, opt-in KV-cache mode for Qwen3.8-27B that applied a normalized
transform to queries and keys, rotated values before four-bit storage, and reversed the value
transform after attention.

It is **not available in this build**. Upstream now owns KV quantization in a dedicated
`kv_cache_append` Op, whose published numerical contract rotates K with a normalized H256
transform and stores V from unrotated BF16 source values. rk8v4's H64 rotation and int4-V
representation have no implementation against that contract yet, so `--kv-dtype rk8v4` parses but
is rejected during engine construction rather than silently degrading to INT8.

Porting RotorQuant onto the new Op is tracked as follow-up work. Until then the previously
published 226,560-token and 247,872-token RotorQuant context measurements do not describe this
build. Use `--kv-dtype int8`, which reaches 171,648 tokens at the 1 GiB headroom boundary.

### Qwen3.8 vision

The same Qwen3.8 artifact supports images. Start the server with `--vision`, MTP3, INT8 KV, and a
32K maximum context. The Windows archive includes `run-qwen38-vision.bat` for this profile.

A 1,920×1,080 image expanded to 2,074 prompt tokens and was read correctly. Measured TTFT was
3.29 seconds, decode reached 98.1 tok/s, MTP acceptance was 96.7%, and startup retained 2.16 GiB
free VRAM. The artifact also declares multi-image and video support; this release test directly
validated a single image.

## Qwen3.6-35B-A3B RTX 3090 results

Measured with the compact 20.84 GiB 35B-A3B artifact, a 4K shared INT8 group-64 paged KV pool,
CUDA Graphs, MTP3, greedy decoding, and no competing GPU workload:

| Concurrent requests | 128 output tokens each | Observed VRAM |
|---:|---:|---:|
| 1 | 162.7 aggregate tok/s | 22,427 MiB |
| 2 | 267.9 aggregate tok/s | 22,743 MiB |
| 4 | 366.2 aggregate tok/s | 23,377 MiB |
| 6 | 383.4 aggregate tok/s | 24,038 MiB |
| 8 | rejected at startup | about 503 MiB over the safe reservation limit |

A longer 512-token-per-request check reached **286.8 tok/s at C1** and **399.1 aggregate tok/s at
C2**. These short-prompt measurements include request-level timing and are not directly comparable
to v0.3.1's 1,500-token adaptive prompt-lookup benchmark.

Compatible-prefix reuse was validated end to end: a repeated 26-token prompt reused 24 tokens,
reducing measured prefill from 371 ms to 10 ms.

### Qwen3.6-35B vision

The compact 35B artifact includes its vision encoder and accepts images through the same OpenAI-
compatible API. Start the server with `--vision` and leave speculative decoding disabled. The
Windows archive includes `run-qwen36-35b-vision.bat` for this profile.

The safe RTX 3090 profile is **one request, 32K maximum context, INT8 KV, vision enabled, and MTP
disabled**. A current v0.6 test processed three 1,920×1,080 images correctly. Each image expanded
to a 2,081-token prompt; engine TTFT was 3.76–4.07 seconds, decode was about 159 tok/s, and peak
VRAM was 23,944 MiB. This leaves little room for another GPU workload.

MTP is intentionally off for this profile. At 32K, speculative recurrent state would exceed the
3090 memory budget; KV compression alone does not recover enough memory. Text-only 35B profiles can
still use MTP3 as documented above.

## Capabilities

- Native SM86 CLI and server applications for Linux and Windows.
- A prebuilt Windows archive with tested launchers.
- OpenAI Chat Completions, Responses, and Anthropic-compatible APIs.
- ReplaySSM and MTP3 for higher throughput without exceeding 24 GB VRAM.
- `low`, `medium`, and `xhigh` reasoning modes.
- Qwen3.8 image understanding with ReplaySSM and MTP3.
- Prefix reuse for faster repeated or shared prompts.
- Qwen3.6-35B image understanding with a guarded 32K profile.
- Windows one-user and eight-user launchers with safe tested defaults.

## Registered model and weight identities

The current source registers these exact `.ninfer` identities. Weight format is not a model
capability guarantee: test artifact identity, target support, and available GPU memory together.

| Model | Weight identity | Notes |
|---|---|---|
| Qwen3.6-27B | `groupwise-int`, `nvfp4` | Text and vision |
| Qwen3.8-27B | `groupwise-int`, `nvfp4` | Text and vision |
| Qwen3.6-35B-A3B | `groupwise-int` | Text/vision; DFlash is text-only |

Model cards and artifact inventories: [Qwen3.6-27B groupwise](model-cards/Qwen3.6-27B-NInfer/README.md),
[Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md),
[Qwen3.8-27B groupwise](model-cards/Qwen3.8-27B-NInfer/README.md),
[Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md), and
[Qwen3.6-35B-A3B](model-cards/Qwen3.6-35B-A3B-NInfer/README.md).

Older container-v1/v2 download-size and measurement notes below refer to named legacy artifacts.
They do not redefine the current registered target/weight identity list above.

## Download and build

The official `.ninfer` model artifacts are published separately; the Windows release archives do
not bundle model weights. Use the five model cards in [Registered model and weight identities](#registered-model-and-weight-identities)
to choose an artifact compatible with the source/runtime. Windows users can also use the model
launcher scripts included in the release archive.

The runtime targets `sm_86`. RTX 3090/3090 Ti are the main measured devices; other Ampere cards may
have less memory, so verify model fit and concurrency before choosing a profile. A recent NVIDIA
driver is required.

Developers can build from source on Windows or Linux. Windows uses Visual Studio 2022 and vcpkg;
Linux uses GCC 13 with system packages or the pinned vcpkg manifest. Both builds require CUDA 12.8
or newer and CMake 3.28 or newer. Linux has no prebuilt release archive; use Docker or build natively.

See the [Windows build guide](docs/rtx-3090-windows.md) or [Linux build guide](docs/rtx-3090-linux.md).
Ordinary Windows release users do not need the development toolchain.

## Qwen3.8 reasoning effort

Qwen3.8-27B supports distinct reasoning-effort modes. `medium` uses the model's normal thinking
prompt. `xhigh` injects the checkpoint's extended deliberation instruction, asking it to validate
assumptions and consider alternatives. This is a real prompt-template change, not a sampling alias.

| Value | Qwen3.8 behavior |
|---|---|
| `none` | Disable thinking |
| `low` | Keep reasoning brief and focused |
| `medium` | Use normal Qwen3.8 thinking |
| `xhigh` | Use extended deliberation and verification |

OpenAI Chat Completions accepts a top-level `reasoning_effort` field:

```json
{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Solve this carefully..."}],
  "reasoning_effort": "xhigh",
  "max_tokens": 4096
}
```

OpenAI Responses uses `"reasoning": {"effort": "xhigh"}`. Anthropic Messages uses
`"output_config": {"effort": "xhigh"}`. For the native CLI, pass
`--reasoning-effort low|medium|xhigh`; use `--no-thinking` instead of an effort to disable
reasoning. Chat Completions returns hidden reasoning separately as `message.reasoning_content`.

## Serving APIs

The server supports:

- OpenAI Chat Completions;
- OpenAI Responses Core with streaming and local continuation state;
- Anthropic Messages;
- compatible-prefix reuse;
- prompt-rendered function tools and parsed tool calls;
- bounded pending-request admission and JSONL request logs.

See [HTTP serving](docs/serving.md) and [CLI usage](docs/cli.md).

## How cohort batching works

The C number is the maximum number of requests NInfer can run together. C1 favors one interactive
user; C8 can combine up to eight active requests into each GPU step for much higher total output.

Follow-up requests do not need to arrive at the same instant. When a running request finishes, the
next waiting request can join at a safe generation boundary. Empty or finished lanes are skipped,
so a C8 server also works normally with only one, two, or four active users.

This is deliberately more bounded than datacenter-style dynamic batching. The maximum number of
users and GPU memory are chosen when the server starts. In return, memory use stays predictable on
a 24 GB card and the server can reuse fast CUDA Graphs instead of rebuilding work continuously.

## Current limits

The runtime targets one CUDA GPU and one resident model; the RTX 3090 is the primary measured device.
- Concurrency is bounded at startup to 1–8 requests. Earlier measurements found compact 35B profiles at C1–C6 and Qwen3.8-27B C8/8K with MTP3; those figures are historical and not a current-build qualification.
- The shared KV pool is fixed at startup and is not divided statically among request lanes.
- This is bounded small-scale batching, not preemptive large-scale continuous batching.
- No multi-GPU execution or CPU weight offload.
- Tool calls are returned to the client but are not executed by NInfer.
- Blackwell-only NVFP4/W4A4 and FP8 A8 tensor-core execution is unavailable on SM86; FP8/NVFP4 weights use supported A16 dequantizing routes.
- The paged runtime exposes BF16 and INT8 group-64 KV; INT8 remains the quality-default path. FP8 E4M3 KV and `rk8v4` are rejected on SM86.

## Current verification

The GitHub Actions workflow on the `ci/github-actions-sm86` update has passed the Python suites and
configured a CUDA 12.8 `sm_86` build that compiled every product and test target. Its GPU CTest job
was skipped because this repository has no self-hosted GPU runner registered. This is compile and
host-test evidence, not a claim that CUDA kernels were executed on a device. Check the
[Actions page](https://github.com/doha-230/ninfer-sm86/actions) for the latest run.

Earlier release gates and device measurements below are retained as historical evidence for the
specific versions, machines, artifacts, and commands stated there. They are not current-tree GPU
qualification.

## Upstream

NInfer-3090 is derived from [Neroued/ninfer](https://github.com/Neroued/ninfer). The upstream project
targets RTX 5090/`sm_120a`; this fork carries the Windows and Linux SM86 compatibility layer,
compact 35B artifact support, and RTX 3090-specific schedules and memory planning.

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the complete, maintained credit list.

- [airtonix](https://github.com/airtonix) added Linux and Docker build and release support in
  [PR #1](https://github.com/Don-Chad/ninfer-3090/pull/1).
- [ColeWheatley](https://github.com/ColeWheatley) contributed SM86 runtime-count/GDN residency
  fixes, ECC diagnostics, and the GeForce-safe Docker fix in
  [PR #7](https://github.com/Don-Chad/ninfer-3090/pull/7).
- [justinlime](https://github.com/justinlime) added NixOS build support in
  [PR #5](https://github.com/Don-Chad/ninfer-3090/pull/5).
- [sry9681](https://github.com/sry9681) contributed the device-wide GPU-memory startup fix in
  [PR #6](https://github.com/Don-Chad/ninfer-3090/pull/6).
- [iamwavecut](https://github.com/iamwavecut) contributed the swscale destination-alignment
  JPEG safety fix in [PR #11](https://github.com/Don-Chad/ninfer-3090/pull/11).
- [nasedkinpv](https://github.com/nasedkinpv) contributed the tool-call parser crash fix in
  [PR #12](https://github.com/Don-Chad/ninfer-3090/pull/12).
- [wmehanna](https://github.com/wmehanna) contributed in-place system-turn rendering for
  Claude Code prefix reuse in [PR #13](https://github.com/Don-Chad/ninfer-3090/pull/13).

## Contributing

Please read the [Pull Request Policy](PR_POLICY.md) before opening an issue or pull request.
It explains how to keep changes focused and how to document correctness, performance, VRAM, and
compatibility evidence.

## License

Apache License 2.0. See [LICENSE](LICENSE).
