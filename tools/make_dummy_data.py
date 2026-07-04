"""
스모크 테스트용 더미 데이터 생성기.

MA-PDM 코드가 기대하는 폴더 구조/파일명 규칙에 맞춰
랜덤 노이즈 프레임(jpg)을 만든다. 이상탐지 성능과는 무관하며,
오직 "학습 파이프라인이 에러 없이 도는가"를 확인하기 위한 용도.

생성 구조 (datasets/addata.py:31 기준):
    <data_dir>/<dataset>/training/frames/<video>/0.jpg, 1.jpg, ...
    <data_dir>/<dataset>/testing/frames/<video>/0.jpg, 1.jpg, ...

파일명 규칙 (addata.py:125,130): 파일명 숫자를 프레임 리스트 인덱스로 쓰므로
반드시 0부터 시작하는 연속 정수여야 한다.

사용 예:
    python tools/make_dummy_data.py --data_dir data --dataset ped2 \
        --image_size 128 --num_videos 2 --frames 20
"""
import argparse
import os
import numpy as np
import cv2


def make_video(frames_dir, num_frames, h, w):
    os.makedirs(frames_dir, exist_ok=True)
    # 파일명은 반드시 zero-padding 해야 한다.
    # addata.py는 frame 리스트를 문자열 .sort()로 정렬한 뒤 "파일명 숫자 == 리스트 인덱스"로 가정한다.
    # 패딩이 없으면 "10.jpg"가 "2.jpg"보다 앞에 정렬되어 인덱스가 어긋나고 IndexError가 난다.
    width = max(4, len(str(num_frames - 1)))
    for k in range(num_frames):
        # 0~255 랜덤 노이즈. cv2는 BGR 순서로 저장하지만 더미라 무관.
        img = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(frames_dir, f"{k:0{width}d}.jpg"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data", help="데이터 루트 (config의 data.data_dir과 동일하게)")
    ap.add_argument("--dataset", default="ped2")
    ap.add_argument("--image_size", type=int, default=128, help="config의 data.image_size와 맞추면 좋음(꼭 같을 필요는 없음, 로더가 resize함)")
    ap.add_argument("--num_videos", type=int, default=2, help="phase별 영상 개수")
    ap.add_argument("--frames", type=int, default=20, help="영상당 프레임 수 (time_step 6보다 충분히 커야 샘플이 생김)")
    args = ap.parse_args()

    h = w = args.image_size
    for phase in ("training", "testing"):
        for v in range(1, args.num_videos + 1):
            frames_dir = os.path.join(args.data_dir, args.dataset, phase, "frames", f"{v:02d}")
            make_video(frames_dir, args.frames, h, w)
            print(f"created {frames_dir} ({args.frames} frames)")

    # 대략적인 학습 샘플 수 안내: 영상당 (frames - time_step) 개
    approx = args.num_videos * max(0, args.frames - 6)
    print(f"\n완료. 학습(training) 샘플 대략 {approx}개 생성됨.")
    print(f"이제 학습 실행: python train_diffusion.py --config {args.dataset}_smoke.yml")


if __name__ == "__main__":
    main()
