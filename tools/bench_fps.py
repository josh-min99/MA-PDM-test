"""
Resolution x FPS microbenchmark for MA-PDM (W1 산출물 #1).

실제 추론 경로(DiffusiveVAD.diffusive_restoration -> generalized_steps_womerge)를
그대로 재사용한다. 합성 텐서를 입력하므로 데이터셋/ckpt 준비 없이 GPU에서 바로 돈다.
(순수 모델 연산 시간만 측정 -> 해상도 스케일링 곡선용. AUC는 여기서 안 잼.)

측정 원칙:
  - batch = 1 (single-stream latency, 실배포 조건)
  - warmup 후 torch.cuda.synchronize() 로 감싼 구간만 계측 (async 편향 제거)
  - grid_r(stride) 고정 -> 해상도만 변수
  - patch_size(64), sampling_timesteps 고정 (W2에서 steps 스윕)

사용 예 (vast RTX 3090):
  python -m tools.bench_fps --config ped2.yml \
      --resolutions 224,256,384,512,720 \
      --sampling_timesteps 5 --grid_r 64 --iters 30 --warmup 10
"""
import argparse
import os
import time
import csv

import numpy as np
import torch
import yaml

import models  # noqa: F401  (register)
from models import DenoisingDiffusion, DiffusiveVAD


def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


def parse_args():
    p = argparse.ArgumentParser(description="MA-PDM resolution x FPS benchmark")
    p.add_argument("--config", type=str, required=True, help="configs/ 안의 yml 파일명")
    p.add_argument("--resume", type=str, default="", help="(선택) ckpt 경로. 없으면 랜덤 가중치로 계측(연산량 동일)")
    p.add_argument("--resolutions", type=str, default="224,256,384,512,720",
                   help="정사각 해상도 목록(콤마 구분). 코드가 imsize x imsize 로 리사이즈함")
    p.add_argument("--sampling_timesteps", type=int, default=5, help="DDIM 역방향 스텝 수")
    p.add_argument("--grid_r", type=int, default=64, help="슬라이딩 스트라이드 r (고정 유지)")
    p.add_argument("--merge", type=str, default="False", help="패치 병합 여부 (False=논문 기본, womerge 경로)")
    p.add_argument("--iters", type=int, default=30, help="계측 반복 횟수")
    p.add_argument("--warmup", type=int, default=10, help="워밍업 반복 횟수(계측 제외)")
    p.add_argument("--out", type=str, default="results/bench", help="CSV/그래프 저장 폴더")
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def build_model(cli, config):
    """DenoisingDiffusion + DiffusiveVAD 를 실제 코드 그대로 세운다."""
    args = argparse.Namespace(
        sampling_timesteps=cli.sampling_timesteps,
        grid_r=cli.grid_r,
        merge=cli.merge,
        image_folder="results/images/",  # test()에서만 쓰임, 여기선 미사용
        resume=cli.resume,
        config=cli.config,
    )
    # ckpt를 여기서 로드하도록 config.sampling.resume 에 덮어씀(있을 때만)
    if cli.resume:
        config.sampling.resume = cli.resume
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveVAD(diffusion, args, config)   # ckpt 없으면 경고만 뜨고 랜덤 가중치로 진행
    diffusion.model.eval()
    return model


def count_patches(imsize, p_size, r):
    n_h = len(range(0, imsize - p_size + 1, r))
    n_w = len(range(0, imsize - p_size + 1, r))
    return n_h * n_w


def bench_one(model, config, imsize, time_step, iters, warmup, r, merge, device):
    p_size = config.data.patch_size
    # x: [b=1, time_step+num_pred(1), 3, H, W]
    x = torch.rand(1, time_step + 1, 3, imsize, imsize, device=device)
    x_cond = x[:, :time_step]
    x_dest = x[:, time_step:]

    for _ in range(warmup):
        model.diffusive_restoration(x_cond, x_dest, r=r, merge=merge)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        model.diffusive_restoration(x_cond, x_dest, r=r, merge=merge)
    torch.cuda.synchronize()
    t1 = time.time()

    latency = (t1 - t0) / iters           # 초/프레임
    fps = 1.0 / latency
    n_patch = count_patches(imsize, p_size, r)
    return latency, fps, n_patch


def main():
    cli = parse_args()
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU가 필요합니다 (generalized_steps_* 가 .to('cuda') 하드코딩). vast에서 실행하세요.")
    device = torch.device("cuda")

    with open(os.path.join("configs", cli.config), "r") as f:
        config = dict2namespace(yaml.safe_load(f))
    config.device = device
    config.sampling.num_diffusion_timesteps = config.sampling.num_diffusion_timesteps  # noqa

    time_step = config.data.time_step
    resolutions = [int(s) for s in cli.resolutions.split(",") if s.strip()]

    model = build_model(cli, config)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"patch_size={config.data.patch_size}  grid_r(stride)={cli.grid_r}  "
          f"ddim_steps={cli.sampling_timesteps}  merge={cli.merge}  batch=1")
    print("-" * 68)
    print(f"{'imsize':>7} | {'#patch':>6} | {'latency(ms)':>11} | {'FPS':>7}")
    print("-" * 68)

    rows = []
    for imsize in resolutions:
        lat, fps, n_patch = bench_one(
            model, config, imsize, time_step,
            cli.iters, cli.warmup, cli.grid_r, cli.merge, device,
        )
        print(f"{imsize:>7} | {n_patch:>6} | {lat*1e3:>11.2f} | {fps:>7.2f}")
        rows.append(dict(imsize=imsize, n_patch=n_patch,
                         latency_ms=round(lat * 1e3, 3), fps=round(fps, 3),
                         ddim_steps=cli.sampling_timesteps, grid_r=cli.grid_r))
    print("-" * 68)

    os.makedirs(cli.out, exist_ok=True)
    tag = f"{config.data.dataset}_s{cli.sampling_timesteps}_r{cli.grid_r}"
    csv_path = os.path.join(cli.out, f"fps_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV -> {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["imsize"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(xs, [r["fps"] for r in rows], "o-", color="tab:blue")
        ax1.set_xlabel("image size (px, square)")
        ax1.set_ylabel("FPS (batch=1)", color="tab:blue")
        ax1.axhline(30, ls="--", lw=0.8, color="gray")
        ax1.text(xs[0], 31, "30 FPS", fontsize=8, color="gray")
        ax2 = ax1.twinx()
        ax2.plot(xs, [r["n_patch"] for r in rows], "s--", color="tab:red", alpha=0.6)
        ax2.set_ylabel("# patches", color="tab:red")
        plt.title(f"MA-PDM {config.data.dataset}  ddim={cli.sampling_timesteps}  r={cli.grid_r}")
        fig.tight_layout()
        png_path = os.path.join(cli.out, f"fps_{tag}.png")
        plt.savefig(png_path, dpi=130)
        print(f"PLOT -> {png_path}")
    except Exception as e:
        print(f"(plot skip: {e})")


if __name__ == "__main__":
    main()
