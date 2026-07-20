#!/bin/bash
# 스텝별 체크포인트 AUC 스윕.
# ddm.py가 스냅샷마다 <dataset>_<step>.pth 를 남기므로, 그 파일들을 스텝 오름차순으로
# 순회하며 eval_diffusion.py로 AUC를 찍고 CSV에 저장, 마지막에 best 스텝을 출력.
#
# 사용: bash tools/sweep_eval.sh [config] [ckpt_dir] [dataset] [sampling_timesteps]
#   예: bash tools/sweep_eval.sh marine.yml data/marine_s2/ckpts marine 5
#
# 주의: config의 sampling.resume 를 매 반복 덮어씀(끝나면 마지막 값이 남음).
#       결과는 results/auc_sweep.csv 에도 저장(스크롤백 날아가도 안전, 유의사항 §8-7).

set -u
CFG=${1:-marine.yml}
CKDIR=${2:-data/marine_s2/ckpts}
DS=${3:-marine}
STEPS=${4:-5}
OUT=results/auc_sweep.csv

mkdir -p results
echo "step,auc" > "$OUT"

# <dataset>_<step>.pth 만 (뒤에 _final 등은 제외), 스텝 숫자 오름차순
mapfile -t CKS < <(ls "$CKDIR"/${DS}_*.pth 2>/dev/null \
  | grep -E "/${DS}_[0-9]+\.pth$" \
  | sort -t_ -k2 -n)

if [ ${#CKS[@]} -eq 0 ]; then
  echo "체크포인트 없음: $CKDIR/${DS}_<step>.pth"
  echo "  (ddm.py 스텝별 저장 패치가 적용된 상태로 학습했는지 확인)"
  exit 1
fi

echo "체크포인트 ${#CKS[@]}개 스윕 시작..."
for f in "${CKS[@]}"; do
  step=$(basename "$f" .pth | sed "s/^${DS}_//")
  abs=$(readlink -f "$f")
  sed -i "s#resume: .*#resume: '$abs'#" "configs/$CFG"
  # AUC는 test_diffusion.py(-> restoration.test())가 계산. eval_diffusion.py는 영상 생성용(AUC 아님).
  auc=$(python test_diffusion.py --config "$CFG" --grid_r 64 \
          --sampling_timesteps "$STEPS" --merge True 2>/dev/null \
        | grep '^AUC:' | grep -oE '[0-9]+\.[0-9]+' | head -1)
  echo "  step $step -> AUC ${auc:-FAIL}"
  echo "$step,${auc:-}" >> "$OUT"
done

echo "=== best (AUC 내림차순 top3) ==="
tail -n +2 "$OUT" | awk -F, 'NF==2 && $2!=""' | sort -t, -k2 -nr | head -3
echo "전체 결과: $OUT"
