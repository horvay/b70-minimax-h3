#!/usr/bin/env bash
# MiniMax H3 Ref2VA on the Arc Pro B70 through stable-diffusion.cpp (Vulkan). No ComfyUI.
# Usage: SCENE=scene-04 W=1344 H=768 FRAMES=90 ATTEMPT=attempt-01 ./run-ref2va.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# Paths: point these at your build and model files.
SD=${SD:-$HOME/ai/stable-diffusion.cpp/build-vulkan/bin/sd-cli}
DIFF=${DIFF:-$HOME/models/dasiwa-hybrid4turbo-v1-q8_0.gguf}
LLM=${LLM:-$HOME/models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf}
VAE=${VAE:-$HOME/models/minimax_h3_video_vae_fp16.safetensors}
AVAE=${AVAE:-$HOME/models/minimax_h3_audio_vae_fp32.safetensors}

SCENE=${SCENE:?SCENE required}
W=${W:-1344}
H=${H:-768}
FRAMES=${FRAMES:?FRAMES required}
STEPS=${STEPS:-4}
SEED=${SEED:-2026100104}
SAMPLER=${SAMPLER:-euler}
ATTEMPT=${ATTEMPT:-attempt-01}
BACKEND=${BACKEND:-diffusion=vulkan0,te=vulkan0,vae=vulkan0}
EXTRA=${EXTRA:-}

# B70: Intel Xe2 coopmat prefill flash attention (patched ggml) in query chunks, and short GPU
# submissions so no job hits the xe 5 s timeout.
export SD_VK_XE_FA=${SD_VK_XE_FA:-1}
export SD_VK_XE_FA_QCHUNK=${SD_VK_XE_FA_QCHUNK:-8192}
export GGML_VK_MAX_NODES_PER_SUBMIT=${GGML_VK_MAX_NODES_PER_SUBMIT:-4}
# Attention offload to the B70 through libtorch-XPU (oneDNN/SYCL kernels, ~5x the Vulkan FA).
# Set SD_XPU_SDPA_LIB= (empty) to fall back to Vulkan attention.
export SD_XPU_SDPA_LIB=${SD_XPU_SDPA_LIB-$HOME/ai/stable-diffusion.cpp/xpu_sdpa/libxpu_sdpa.so}
[[ -z "$SD_XPU_SDPA_LIB" ]] && unset SD_XPU_SDPA_LIB
# Transformer blocks on the XPU through libtorch (weights from the original int8 convrot checkpoint).
# Set SD_XPU_H3_LIB= (empty) to keep the blocks on Vulkan.
export SD_XPU_H3_LIB=${SD_XPU_H3_LIB-$HOME/ai/stable-diffusion.cpp/xpu_sdpa/libxpu_h3.so}
export SD_XPU_H3_WEIGHTS=${SD_XPU_H3_WEIGHTS:-$HOME/models/DasiwaMinimaxH3_dasiwaHybrid4turboV1.safetensors}
[[ -z "$SD_XPU_H3_LIB" ]] && unset SD_XPU_H3_LIB
# Vulkan VRAM budget: with SD_XPU_H3_LIB, torch loads ~19.5 GB of block weights first; Vulkan then
# budgets against live free VRAM (0) and fits the text encoder and VAE into the rest.
MAX_VRAM=${MAX_VRAM:-$([[ -n "${SD_XPU_H3_LIB:-}" ]] && echo 0 || { [[ -n "${SD_XPU_SDPA_LIB:-}" ]] && echo 25 || echo 0; })}
# Patched Mesa ANV (main + MR !43384 DPAS register coalescing), loaded only for this process.
# Only the Intel ICD is visible, so the B70 is vulkan0.
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-$HOME/mesa-b70/install/share/vulkan/icd.d/intel_icd.x86_64.json}

DIR="$HERE/$SCENE"
IN="$DIR/inputs"
OUT="$DIR/$ATTEMPT"
mkdir -p "$OUT"

refs=()
for f in "$IN"/ref-*.png; do refs+=(-r "$f"); done
videos=()
[[ -d "$IN/motion-frames" ]] && videos+=(--ref-video "$IN/motion-frames")
[[ -f "$IN/motion-audio.wav" ]] && videos+=(--ref-video-audio "$IN/motion-audio.wav")
# Keyframe guides (patched sd-cli): inputs/guides.txt lines are FILE@FRAME[@STRENGTH], FILE relative to inputs/.
guides=()
if [[ -f "$IN/guides.txt" ]]; then
  while read -r spec; do
    [[ -z "$spec" || "$spec" == \#* ]] && continue
    guides+=(--guide "$IN/$spec")
  done < "$IN/guides.txt"
  cp "$IN/guides.txt" "$OUT/guides.txt"
fi
[[ -f "$IN/guide-audio.wav" ]] && guides+=(--guide-audio "$IN/guide-audio.wav@0")

cp "$DIR/prompt.txt" "$OUT/prompt.txt"
cat > "$OUT/settings.env" <<EOF
SCENE=$SCENE W=$W H=$H FRAMES=$FRAMES STEPS=$STEPS SEED=$SEED SAMPLER=$SAMPLER
BACKEND=$BACKEND
DIFF=$DIFF
LLM=$LLM
EXTRA=$EXTRA
SD_XPU_H3_LIB=${SD_XPU_H3_LIB:-} SD_XPU_SDPA_LIB=${SD_XPU_SDPA_LIB:-} MAX_VRAM=$MAX_VRAM VK_ICD_FILENAMES=$VK_ICD_FILENAMES SD_VK_XE_FA=$SD_VK_XE_FA SD_VK_XE_FA_QCHUNK=$SD_VK_XE_FA_QCHUNK GGML_VK_MAX_NODES_PER_SUBMIT=$GGML_VK_MAX_NODES_PER_SUBMIT
started=$(date -Iseconds)
EOF

# shellcheck disable=SC2086
"$SD" -M vid_gen \
  --diffusion-model "$DIFF" \
  --vae "$VAE" \
  --audio-vae "$AVAE" \
  --llm "$LLM" \
  --backend "$BACKEND" \
  --offload-to-cpu \
  ${FA_FLAG:---fa} \
  "${refs[@]}" "${videos[@]}" "${guides[@]}" \
  -p "$(cat "$DIR/prompt.txt")" \
  --cfg-scale 1 \
  --sampling-method "$SAMPLER" \
  --scheduler simple \
  --steps "$STEPS" \
  --flow-shift 12 \
  -W "$W" -H "$H" \
  --video-frames "$FRAMES" \
  --fps 24 \
  --rng cpu \
  --seed "$SEED" \
  -o "$OUT/raw.mp4" \
  --max-vram "$MAX_VRAM" \
  -v $EXTRA 2>&1 | tee "$OUT/run.log"

echo "finished=$(date -Iseconds)" >> "$OUT/settings.env"
