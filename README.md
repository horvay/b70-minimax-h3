# MiniMax-H3 video generation on an Intel Arc Pro B70

Native 1344x768 MiniMax-H3 video (a 20B-parameter joint audio/video diffusion
transformer) on one Intel Arc Pro B70 (Battlemage, 32 GB) under Linux, without
ComfyUI. Running it at a usable speed meant fixing several separate bottlenecks.
This repo records them along with the patches, tools and numbers.

**Result:** one denoising step of a 1344x768, 107-frame Ref2VA job (~50k tokens)
went from **~570 s to ~71 s**, and a whole clip from **42 min to ~9 min**.

https://github.com/horvay/b70-minimax-h3/raw/main/demo/scene05-cartoon-vs-b70-1344x768.mp4

`demo/scene05-cartoon-vs-b70-1344x768.mp4` is a reference remake: the source
animation on the left and the B70 render on the right, guided by realistic stills
and the cartoon's motion.

## Where the code is

- **stable-diffusion.cpp fork:** [horvay/stable-diffusion.cpp `b70-xpu`](https://github.com/horvay/stable-diffusion.cpp/tree/b70-xpu).
  Build and run notes are in `docs/b70_xpu.md`.
- **ggml fork (Vulkan kernels):** [horvay/ggml `b70-xpu`](https://github.com/horvay/ggml/tree/b70-xpu). It is the submodule of the branch above.
- `patches/` holds the same changes as plain patches against upstream:
  - `stable-diffusion.cpp.patch` applies to leejet/stable-diffusion.cpp `3f8527a`.
  - `ggml.patch` applies to leejet/ggml `89c4413`.
  - `mesa-mr43384-dpas-coalesce.patch` is Mesa MR !43384 (brw DPAS register coalescing).
- `xpu/` holds the libtorch-XPU helper libraries (`xpu_h3.cpp`, `xpu_sdpa.cpp`) and `build.sh`.
- `tools/` contains:
  - `unconvrot.py`: turns a ComfyUI int8 "convrot" checkpoint into bf16 for GGUF requantization.
  - `gpu-cpu-monitor.ts`: Bun script that samples xe engine busy % and CPU % and logs the GPU/CPU ping-pong.
  - `run-ref2va.sh`: the runner used for the renders.

## The bottlenecks, in the order they were hit

| # | Symptom | Cause | Fix | Effect |
| --- | --- | --- | --- | --- |
| 1 | `device lost` on the first step | ggml Vulkan FA falls back to a scalar shader on Xe2 (ANV only exposes 8x16x16 coopmat). One FA dispatch outlives the xe driver's 5 s job timeout | Port llama.cpp PR #29357 Intel prefill FA, add a mask-less variant, chunk queries, `GGML_VK_MAX_NODES_PER_SUBMIT=4` | 3.5 → 10 TFLOPS, no timeouts |
| 2 | ~7 TOPS matmuls | sd.cpp's int8 *convrot* Vulkan matmul is untuned on Intel | Undo the Hadamard rotation and requantize to Q8_0 (`tools/unconvrot.py` + `sd-cli -M convert`) | 30-60 TFLOPS |
| 3 | GPU/CPU ping-pong, ~70 min/step | MLP `fc1` output (5 GB) exceeds the 4 GiB Vulkan buffer limit, so the scheduler silently runs every MLP on the CPU | Token-chunk MLP and QKV (`SD_H3_TOKEN_CHUNK`) | ~6.8 min/step |
| 4 | FA stuck at ~10-14 TFLOPS in every Vulkan kernel | Mesa ANV compiles Xe2 compute at 128 GRF; DPAS kernels spill heavily | Mesa MR !43384 and tuned geometry for unmasked FA | ~18 TFLOPS |
| 5 | Attention still ~75% of the step | Vulkan cannot reach oneDNN's attention throughput (~95 TFLOPS on the same card, with 256-GRF SYCL kernels) | Run all H3 transformer blocks on the XPU via libtorch, with weights from the original int8 checkpoint | **~71 s/step** |

PyTorch XPU on this card measured ~95 TFLOPS for SDPA at 56 heads x 16k tokens and
100-140 TFLOPS for bf16 GEMMs at H3's MLP shapes. The best Vulkan FA was 18 TFLOPS.

## Gotchas worth knowing

- **xe job timeout:** 5 s by default, at most 10 s via root sysfs. Long dispatches die with "device lost".
- **4 GiB Vulkan storage-buffer limit:** ops above it run on the CPU without any warning.
  `SD_LOG_UNSUPPORTED_OPS=1` prints them.
- **Two GPU runtimes in one process.** Vulkan (ggml) and Level Zero (torch) share 32 GB:
  - Vulkan should budget against live free memory (`--max-vram 0`).
  - Release idle Vulkan memory before sampling, load torch weights lazily, and free them before VAE decode.
- **Ref2VA reference videos** are VAE-encoded in one piece, so keep them small (448x256).
  They are not frame-locked to the output: they carry how she moves, not exactly when.
  Light keyframe guides (`--guide img@frame@0.3`) fix the timing; full-strength guides cause stop-go motion.
- **H3 VAE decode** with tiles above 256 px or overlap below 0.25 produces vertical streaks.
- **Test small, at realistic shapes.** A 64-keys-per-pass FA config passed every small unit test but
  corrupted output at 56 heads. A 512x288, 1-step probe render catches this in about a minute.

## Hardware and software

Arc Pro B70 32 GB, Ryzen 9 5900XT, 62 GB RAM, Linux 7.x, Mesa 26.2 (plus the MR above,
built ANV-only and loaded through `VK_ICD_FILENAMES`), intel-compute-runtime 26.31,
PyTorch 2.11+xpu. Model: DaSiWa MiniMax-H3 Hybrid 4Turbo (int8 convrot), 4 steps,
euler/simple, flow shift 12.
