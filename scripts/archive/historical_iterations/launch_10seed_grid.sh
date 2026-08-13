#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
REGIONS="india cambodia vietnam kenya france netherlands"
NEW_SEEDS="3 4 5 6 7 8 9"
(
  export CUDA_VISIBLE_DEVICES=0
  echo "[GPU0 START] $(date)" > /tmp/_v8_gpu0.log
  for S in $NEW_SEEDS; do
    for R in $REGIONS; do
      for MODE in backbone none; do
        TAG="prithvi_v8_${R}_${MODE}_s${S}"
        OUT="data/results/ftw_finetune_fm_prithvi${TAG}.json"
        if [ -f "$OUT" ]; then
          echo "[GPU0 SKIP $(date +%H:%M:%S)] $TAG" >> /tmp/_v8_gpu0.log
          continue
        fi
        echo "[GPU0 RUN  $(date +%H:%M:%S)] $TAG" >> /tmp/_v8_gpu0.log
        .venv/bin/python scripts/ftw_finetune_fm.py --model prithvi --freeze $MODE --seed $S --epochs 80 --tag "$TAG" "$R" 2>&1 | tail -2 >> /tmp/_v8_gpu0.log
      done
    done
  done
  echo "[GPU0 END] $(date)" >> /tmp/_v8_gpu0.log
) &
GPU0_PID=$!
(
  export CUDA_VISIBLE_DEVICES=1
  echo "[GPU1 START] $(date)" > /tmp/_v8_gpu1.log
  for S in $NEW_SEEDS; do
    for R in $REGIONS; do
      for MODE in backbone none; do
        TAG="terramind_v8_${R}_${MODE}_s${S}"
        OUT="data/results/ftw_finetune_fm_terramind${TAG}.json"
        if [ -f "$OUT" ]; then
          echo "[GPU1 SKIP $(date +%H:%M:%S)] $TAG" >> /tmp/_v8_gpu1.log
          continue
        fi
        echo "[GPU1 RUN  $(date +%H:%M:%S)] $TAG" >> /tmp/_v8_gpu1.log
        .venv/bin/python scripts/ftw_finetune_fm.py --model terramind --freeze $MODE --seed $S --epochs 80 --tag "$TAG" "$R" 2>&1 | tail -2 >> /tmp/_v8_gpu1.log
      done
    done
  done
  echo "[GPU1 END] $(date)" >> /tmp/_v8_gpu1.log
) &
GPU1_PID=$!
echo "Launched: 168 runs (84 per GPU). PIDs: $GPU0_PID $GPU1_PID"
wait $GPU0_PID $GPU1_PID
echo "[ALL DONE] $(date)" | tee -a /tmp/_v8_grid.log
