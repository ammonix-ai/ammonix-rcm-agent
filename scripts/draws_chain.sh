#!/usr/bin/env bash
# The measurements under the payer's adjudication, unattended:
#   three draws (tranches 99, 100, 101), every arm writing through the tuned
#   writer; then the L-seat swap (four writers on the demonstration claims).
# Requires the fine-tune chain to have created container ammonix-ornith-ft.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
LOG=runs/reports/draws_chain.log
ARMS=${ARMS_OVERRIDE:-personas,ammonix,agentic,agentic_rag,claude,sol_tuned,sol_rag_tuned}

for T in 99 100 101; do
  echo "[draws] tranche $T $(date)" | tee -a $LOG
  $PY scripts/rollout_paperwork.py --tranche $T --arms $ARMS --writer ornith_ft \
      --report comparison_paperwork_t$T.json > runs/reports/rollout_paperwork_t$T.log 2>&1
  echo "[draws] tranche $T exit $? $(date)" | tee -a $LOG
done

up() {  # up <container> <port> <served-substring>
  docker ps --format '{{.Names}}' | grep -v "^$1$" | xargs -r docker stop >/dev/null 2>&1 || true
  docker start "$1" >/dev/null
  for i in $(seq 1 90); do sleep 10; curl -s "http://127.0.0.1:$2/v1/models" | grep -q "$3" && return 0; done
  echo "server $1 did not come up" >&2; return 1
}

echo "[lswap] qwen $(date)" | tee -a $LOG
up ammonix-m1-27b 8000 Qwen
$PY scripts/l_swap_experiment.py --model qwen --workers 6 > runs/reports/l_swap_qwen.log 2>&1
echo "[lswap] qwen exit $?" | tee -a $LOG
echo "[lswap] ornith $(date)" | tee -a $LOG
up ammonix-ornith-9b 8002 '"Ornith-1.5-9B"'
$PY scripts/l_swap_experiment.py --model ornith --workers 6 > runs/reports/l_swap_ornith.log 2>&1
echo "[lswap] ornith exit $?" | tee -a $LOG
echo "[lswap] ornith_ft $(date)" | tee -a $LOG
up ammonix-ornith-ft 8002 rcm-r2
$PY scripts/l_swap_experiment.py --model ornith_ft --workers 6 > runs/reports/l_swap_ornith_ft.log 2>&1
echo "[lswap] ornith_ft exit $?" | tee -a $LOG
echo "[lswap] opus5 $(date)" | tee -a $LOG
$PY scripts/l_swap_experiment.py --model opus5 --workers 4 > runs/reports/l_swap_opus5.log 2>&1
echo "[lswap] opus5 exit $?" | tee -a $LOG
$PY scripts/l_swap_experiment.py --compare > runs/reports/l_swap_compare.log 2>&1
echo "[draws] all done $(date)" | tee -a $LOG
