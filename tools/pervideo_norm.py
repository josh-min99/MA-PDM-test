# -*- coding: utf-8 -*-
"""per-video 정규화 + 점수집계(frame-mean vs patch-max) 진단 (GPU 불필요).

test()가 저장한 results/images/<ds>/<ds>.pkl 에서 두 종류의 프레임 점수를 읽는다:
  - img_score  : 프레임 전체 MSE '평균'  ← test()의 AUC가 실제로 쓰는 값
  - patch_score: 패치별 오차의 '최댓값'  ← 저장만 되고 AUC엔 미사용
작은 객체(군함 ~13x5px @256^2)는 frame-mean에 희석되므로, patch-max로 재보면
"신호가 있는데 평균이 죽였나"를 재학습 없이 판별할 수 있다.

각 점수원마다:
  변형별 전역 MicroAUC (raw / z_offline / z_causal / minmax)
  [A] test() MacroAUC 재현(모든영상+sentinel) — 0.99가 조작인지
  [B] 정직한 per-video AUC (정상·군함 섞인 영상만, sentinel 없음)
  [C] 혼합 영상 프레임만 pooled

filt/gaussian_filter는 models/utils.py에서 그대로 복사(검증됨).
"""
import argparse
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc


def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    filt = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filt


def filt(input, range=302, mu=21):
    filter_2d = gaussian_filter(np.arange(1, range), mu)
    frame_scores = np.asarray(input, dtype=np.float64)
    pad = len(filter_2d) // 2
    in_ = np.concatenate((np.zeros(pad), frame_scores, np.zeros(pad)))
    return np.nan_to_num(np.correlate(in_, filter_2d, 'valid'), nan=0.)


def z_offline(s):
    s = np.asarray(s, float)
    return (s - s.mean()) / (s.std() + 1e-8)


def z_causal_expanding(s):
    s = np.asarray(s, float); out = np.zeros_like(s)
    for t in range(len(s)):
        w = s[:t + 1]; out[t] = (s[t] - w.mean()) / (w.std() + 1e-8)
    return out


def z_causal_window(s, W):
    s = np.asarray(s, float); out = np.zeros_like(s)
    for t in range(len(s)):
        w = s[max(0, t - W + 1):t + 1]; out[t] = (s[t] - w.mean()) / (w.std() + 1e-8)
    return out


def minmax_offline(s):
    s = np.asarray(s, float)
    return (s - s.min()) / (s.max() - s.min() + 1e-8)


def build(per_video, tf=None):
    return np.concatenate([tf(s) if tf else np.asarray(s, float) for s in per_video])


def analyze(score_dict, videos_list, labels_list, title, args):
    print("\n" + "=" * 66)
    print(f"### 점수원: {title}")
    print("=" * 66)
    order = sorted(videos_list)
    per_video = [filt(score_dict[v.replace('\\', '/').split('/')[-1]],
                      range=args.range, mu=args.mu) for v in order]
    total = sum(len(s) for s in per_video)
    if total != len(labels_list):
        print(f"  길이 불일치 {total} vs {len(labels_list)} — 건너뜀"); return

    variants = [
        ("raw", None),
        ("z_offline", z_offline),
        ("z_causal_expanding", z_causal_expanding),
        (f"z_causal_window(W={args.window})", lambda s: z_causal_window(s, args.window)),
        ("minmax_offline", minmax_offline),
    ]
    print(f"{'variant':30} {'MicroAUC':>9} {'AP':>8}")
    print("-" * 50)
    for name, tf in variants:
        sc = build(per_video, tf)
        print(f"{name:30} {roc_auc_score(labels_list, sc):9.4f} {_ap(sc, labels_list):8.4f}")

    # per-video 라벨 분할
    pvl, off = [], 0
    for s in per_video:
        pvl.append(labels_list[off:off + len(s)]); off += len(s)

    # [A] macro sentinel 재현
    macro = []
    for s, lab in zip(per_video, pvl):
        macro.append(roc_auc_score(np.concatenate(([0], lab, [1])),
                                   np.concatenate(([0], s, [1]))))
    print(f"[A] MacroAUC(sentinel 재현): {np.nanmean(macro):.4f}")

    # [B] 정직한 per-video (혼합만)
    honest, nb, npos, nneg = [], 0, 0, 0
    for s, lab in zip(per_video, pvl):
        u = set(int(x) for x in np.unique(lab))
        if 0 in u and 1 in u:
            nb += 1; honest.append(roc_auc_score(lab, s))
        elif u == {1}: npos += 1
        elif u == {0}: nneg += 1
    print(f"[B] 정직한 per-video AUC (혼합 {nb}개 / 전부군함 {npos} / 전부정상 {nneg})")
    if honest:
        h = np.array(honest)
        print(f"    mean {h.mean():.4f} / median {np.median(h):.4f} / "
              f">0.7 {int((h > 0.7).sum())}/{len(h)}")

    # [C] 혼합 프레임 pooled
    mr, mz, ml = [], [], []
    for s, lab in zip(per_video, pvl):
        u = set(int(x) for x in np.unique(lab))
        if 0 in u and 1 in u:
            mr.append(np.asarray(s, float)); mz.append(z_offline(s)); ml.append(np.asarray(lab))
    if ml:
        L = np.concatenate(ml)
        print(f"[C] 혼합프레임 pooled ({len(L)}f, 군함 {100*L.mean():.0f}%): "
              f"raw {roc_auc_score(L, np.concatenate(mr)):.4f} / "
              f"z {roc_auc_score(L, np.concatenate(mz)):.4f}")


def _ap(scores, labels):
    p, r, _ = precision_recall_curve(labels, scores)
    return sk_auc(r, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--range", type=int, default=38)
    ap.add_argument("--mu", type=int, default=11)
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        d = pickle.load(f)
    labels = np.asarray(d["labels_list"]).ravel()
    vids = d["videos_list"]
    print(f"영상 {len(vids)} / 라벨 {len(labels)} / 이상비율 {100*labels.mean():.1f}%")

    analyze(d["img_score"], vids, labels, "img_score (frame-mean, test 사용)", args)
    if "patch_score" in d:
        analyze(d["patch_score"], vids, labels, "patch_score (patch-max, 미사용)", args)

    print("\n해석:")
    print(" - patch_score의 [B]/[C]가 img_score보다 크게 높으면 →")
    print("   '신호는 있는데 frame-mean 평균이 죽였다'(작은객체 희석) → patch-max로 바꾸면 됨(재학습 불필요).")
    print(" - patch_score도 ~0.5면 → 모델이 군함을 인코딩 못함 → 해상도/과적합 손봐야 함.")


if __name__ == "__main__":
    main()
