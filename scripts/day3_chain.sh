#!/usr/bin/env bash
# After the ablation: the imitation arm on the three draws, the demo-UI
# checker, and the demo data bundle.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
LOG=runs/reports/day3_chain.log

until grep -q "^exit" runs/reports/rollout_ablation_t99.log 2>/dev/null; do sleep 60; done
echo "[day3] ablation finished: $(grep '^exit' runs/reports/rollout_ablation_t99.log) $(date)" | tee -a $LOG

for T in 99 100 101; do
  echo "[day3] imitation tranche $T $(date)" | tee -a $LOG
  $PY scripts/rollout_paperwork.py --tranche $T --arms imitation,oracle --writer ornith_ft \
      --report extra_arms_paperwork_t$T.json > runs/reports/rollout_extra_t$T.log 2>&1
  echo "[day3] imitation tranche $T exit $? $(date)" | tee -a $LOG
done

echo "[day3] demo-ui check $(date)" | tee -a $LOG
docker ps --format '{{.Names}}' | grep -v '^ammonix-m1-27b$' | xargs -r docker stop >/dev/null 2>&1 || true
docker start ammonix-m1-27b >/dev/null 2>&1 || true
for i in $(seq 1 90); do sleep 10; curl -s http://127.0.0.1:8000/v1/models | grep -q Qwen && break; done
$PY scripts/check_demo_ui.py > runs/reports/demo_ui_check_v7.log 2>&1
echo "[day3] demo-ui check exit $? $(date)" | tee -a $LOG

echo "[day3] package bundle $(date)" | tee -a $LOG
$PY scripts/package_demo_data.py > runs/reports/package_demo_v7.log 2>&1
echo "[day3] package exit $? $(date)" | tee -a $LOG
echo "[day3] all done $(date)" | tee -a $LOG
