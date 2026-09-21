#!/usr/bin/env bash
# One unattended fine-tuning round on the payer's reward:
# prompts -> sample (base Ornith served) -> score (scrubber + payer) -> select
# -> train (no server) -> merge. Every stage is resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
TPY="${ORNITH_TRAIN_PY:-python}"  # python of a venv with torch + trl
export PYTHONIOENCODING=utf-8
LOG=runs/sft/ornith_r2/chain.log
mkdir -p runs/sft/ornith_r2

up() {  # up <container> <port> <served-substring>
  docker ps --format '{{.Names}}' | grep -v "^$1$" | xargs -r docker stop >/dev/null 2>&1 || true
  docker start "$1" >/dev/null
  for i in $(seq 1 90); do sleep 10; curl -s "http://127.0.0.1:$2/v1/models" | grep -q "$3" && return 0; done
  echo "server $1 did not come up" >&2; return 1
}

echo "[chain] prompts $(date)" | tee -a $LOG
$PY scripts/l_finetune_ornith.py prompts --per-skill-cap 200 >> $LOG 2>&1

echo "[chain] sample $(date)" | tee -a $LOG
up ammonix-ornith-9b 8002 '"Ornith-1.5-9B"'
$PY scripts/l_finetune_ornith.py sample --k 8 --temperature 0.8 --workers 8 >> $LOG 2>&1

echo "[chain] score $(date)" | tee -a $LOG
$PY scripts/l_finetune_ornith.py score --workers 6 >> $LOG 2>&1

echo "[chain] select $(date)" | tee -a $LOG
$TPY scripts/l_finetune_ornith.py select >> $LOG 2>&1

echo "[chain] train $(date)" | tee -a $LOG
docker ps --format '{{.Names}}' | xargs -r docker stop >/dev/null 2>&1 || true
sleep 10
$TPY scripts/l_finetune_ornith.py train --tag r2 --epochs 2 >> $LOG 2>&1

echo "[chain] merge $(date)" | tee -a $LOG
$TPY scripts/l_finetune_ornith.py merge --tag r2 --out-name Ornith-1.5-9B-rcm-r2 >> $LOG 2>&1
echo "[chain] done $(date)" | tee -a $LOG

echo "[chain] serve r2 $(date)" | tee -a $LOG
docker rm -f ammonix-ornith-ft >/dev/null 2>&1 || true
MSYS_NO_PATHCONV=1 docker create --name ammonix-ornith-ft --gpus all --shm-size 1g -p 8002:8000 \
  -v "${MODELS_DIR:-$HOME/models}":/models -v ammonix-hf:/root/.cache/huggingface vllm/vllm-openai:v0.19.1 \
  --model /models/Ornith-1.5-9B-rcm-r2 --served-model-name Ornith-1.5-9B-rcm-r2 --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --enforce-eager --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 --trust-remote-code >/dev/null
echo "[chain] container created $(date)" | tee -a $LOG
