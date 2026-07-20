# -*- coding: utf-8 -*-
"""per-video 정규화가 MicroAUC를 살리는지 검증 (GPU 불필요, pkl만 필요).

test()가 저장한 results/images/<ds>/<ds>.pkl 에서 프레임별 점수를 읽어,
restoration.test()와 동일하게 이상점수를 재현한 뒤(= filt(img_score)),
정규화 방식별 전역 MicroAUC를 비교한다:
  1) raw            : 원본(정규화 없음) — test()가 낸 값 재현(검증용)
  2) z_offline      : 영상별 z-score (mean/std, 전체프레임=비인과적)
  3) z_causal_exp   : 영상별 인과적 z (확장 윈도우, 배포가능)
  4) z_causal_win W : 영상별 인과적 z (슬라이딩 윈도우)
  5) minmax_offline : 영상별 min-max (VAD 관행)

가설: raw는 낮은데(0.13) z_offline이 뛰면 → "기준선 차이가 원인" 증명.
      z_causal도 뛰면 → 스트리밍에서도 됨(강한 결과). 안 뛰면 → 스트리밍 정규화가 연구문제.

filt/gaussian_filter는 models/utils.py에서 그대로 복사(토치 의존 제거, 어디서나 실행).
"""
import argparse
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc


# --- models/utils.py 에서 그대로 복사 (재현 충실성) ---
def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    filt = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filt


def filt(input, range=302, mu=21):
    filter_2d = gaussian_filter(np.arange(1, range), mu)
    frame_scores = np.asarray(input, dtype=np.float64)
    padding_size = len(filter_2d) // 2
    in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    frame_scores = np.correlate(in_, filter_2d, 'valid')
    return np.nan_to_num(frame_scores, nan=0.)
# ---


def z_offline(s):
    s = np.asarray(s, float)
    sd = s.std()
    return (s - s.mean()) / (sd + 1e-8)


def z_causal_expanding(s):
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    for t in range(len(s)):
        w = s[:t + 1]
        out[t] = (s[t] - w.mean()) / (w.std() + 1e-8)
    return out


def z_causal_window(s, W):
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    for t in range(len(s)):
        w = s[max(0, t - W + 1):t + 1]
        out[t] = (s[t] - w.mean()) / (w.std() + 1e-8)
    return out


def minmax_offline(s):
    s = np.asarray(s, float)
    rng = s.max() - s.min()
    return (s - s.min()) / (rng + 1e-8)


def micro_auc(scores, labels):
    return roc_auc_score(labels, scores)


def micro_ap(scores, labels):
    p, r, _ = precision_recall_curve(labels, scores)
    return sk_auc(r, p)


def build(per_video_scores, transform=None):
    """정렬된 영상별 점수를 정규화(옵션) 후 이어붙임."""
    out = []
    for s in per_video_scores:
        out.append(transform(s) if transform else np.asarray(s, float))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--range", type=int, default=38)   # test() 호출값
    ap.add_argument("--mu", type=int, default=11)
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    img_score = d["img_score"]          # {video_name: [frame scores]}
    videos_list = d["videos_list"]
    labels_list = np.asarray(d["labels_list"]).ravel()

    # test()와 동일 순서: sorted(videos_list), 이상점수 = filt(img_score[name])
    order = sorted(videos_list)
    per_video = []
    for v in order:
        name = v.split("/")[-1]
        per_video.append(filt(img_score[name], range=args.range, mu=args.mu))

    total = sum(len(s) for s in per_video)
    print(f"영상 {len(order)}개 / 점수프레임 {total} / 라벨 {len(labels_list)} "
          f"/ 정합 {total == len(labels_list)}")
    print(f"이상(1) 비율 {100*labels_list.mean():.1f}%\n")
    if total != len(labels_list):
        raise SystemExit("점수-라벨 길이 불일치 — order/키 확인 필요")

    variants = [
        ("raw (원본, test 재현)", None),
        ("z_offline (비인과)", z_offline),
        ("z_causal_expanding (배포가능)", z_causal_expanding),
        (f"z_causal_window(W={args.window})", lambda s: z_causal_window(s, args.window)),
        ("minmax_offline", minmax_offline),
    ]
    print(f"{'variant':32} {'MicroAUC':>9} {'AP':>8}")
    print("-" * 52)
    for name, tf in variants:
        scores = build(per_video, tf)
        a = micro_auc(scores, labels_list)
        p = micro_ap(scores, labels_list)
        print(f"{name:32} {a:9.4f} {p:8.4f}")

    # ---- 영상별 라벨 분할 (raw 재현됨 → 정렬 정확) ----
    per_video_labels, off = [], 0
    for s in per_video:
        per_video_labels.append(labels_list[off:off + len(s)])
        off += len(s)

    # ---- (A) test() macro 재현: 모든 영상 + sentinel([0],[1]) ----
    macro_sentinel = []
    for s, lab in zip(per_video, per_video_labels):
        lbl2 = np.concatenate(([0], lab, [1]))
        prd2 = np.concatenate(([0], s, [1]))
        macro_sentinel.append(roc_auc_score(lbl2, prd2))
    print(f"\n[A] test() MacroAUC 재현(모든영상+sentinel): {np.nanmean(macro_sentinel):.4f}"
          f"  (원output 0.9927와 비교 → 일치하면 'macro는 sentinel 조작' 확인)")

    # ---- (B) 정직한 per-video AUC: 혼합 영상만, sentinel 없음 ----
    honest, n_both, n_pos, n_neg = [], 0, 0, 0
    for s, lab in zip(per_video, per_video_labels):
        u = set(int(x) for x in np.unique(lab))
        if 0 in u and 1 in u:
            n_both += 1
            honest.append(roc_auc_score(lab, s))
        elif u == {1}:
            n_pos += 1
        elif u == {0}:
            n_neg += 1
    print(f"\n[B] 정직한 per-video AUC (정상·군함 섞인 영상만, sentinel 없음)")
    print(f"    혼합영상 {n_both}개 / 전부군함 {n_pos} / 전부정상 {n_neg}")
    if honest:
        h = np.array(honest)
        print(f"    per-video AUC: mean {h.mean():.4f} / median {np.median(h):.4f} "
              f"/ min {h.min():.4f} / max {h.max():.4f}")
        print(f"    >0.7 인 영상 {int((h>0.7).sum())}/{len(h)} , <0.5 인 영상 {int((h<0.5).sum())}/{len(h)}")

    # ---- (C) 혼합 영상 프레임만 모아 pooled AUC (raw / z_offline) ----
    mix_raw, mix_z, mix_lab = [], [], []
    for s, lab in zip(per_video, per_video_labels):
        u = set(int(x) for x in np.unique(lab))
        if 0 in u and 1 in u:
            mix_raw.append(np.asarray(s, float))
            mix_z.append(z_offline(s))
            mix_lab.append(np.asarray(lab))
    if mix_lab:
        ml = np.concatenate(mix_lab)
        print(f"\n[C] 혼합영상 프레임만 pooled ({len(ml)}프레임, 군함 {100*ml.mean():.0f}%)")
        print(f"    raw MicroAUC      {roc_auc_score(ml, np.concatenate(mix_raw)):.4f}")
        print(f"    z_offline MicroAUC {roc_auc_score(ml, np.concatenate(mix_z)):.4f}")

    print("\n해석:")
    print(" - [A]가 0.99 재현 → macro는 sentinel 조작(무의미).")
    print(" - [B]/[C]가 높으면(0.7+) → 같은 장면 안에선 군함 잡음 → 문제는 테스트설계 → spot2가 해결.")
    print(" - [B]/[C]도 ~0.5면 → 진짜 못 잡음 → 크기(256²서 ~13×5px)·과적합 원인 → spot2로 안 풀림.")


if __name__ == "__main__":
    main()
