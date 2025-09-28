import os, cv2, numpy as np, random, argparse

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def make_clip(mode, out_path, seconds=1.5, fps=16, size=224, box_size=60, seed=0, codec="mp4v"):
    rng = random.Random(seed)
    frames = int(seconds * fps)
    H = W = size

    # background + box color
    bg_val = rng.randint(160, 220)
    bg = np.full((H, W, 3), bg_val, dtype=np.uint8)
    color = (rng.randint(20, 80), rng.randint(20, 80), rng.randint(20, 80))

    # path of motion
    y = rng.randint(40, H - 40 - box_size)
    if mode == "swipe_left":
        x_start, x_end = W - 20 - box_size, 20
    elif mode == "swipe_right":
        x_start, x_end = 20, W - 20 - box_size
    elif mode == "stop":
        x_start = x_end = (W - box_size) // 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter with codec '{codec}'. "
            "Try --codec XVID and use .avi extension, e.g. out.avi"
        )

    for t in range(frames):
        alpha = t / max(1, frames - 1)
        x = int((1 - alpha) * x_start + alpha * x_end)
        # small jitter to avoid being too synthetic
        jitter_x, jitter_y = rng.randint(-2, 2), rng.randint(-2, 2)
        frame = bg.copy()
        cv2.rectangle(frame, (x + jitter_x, y + jitter_y),
                      (x + jitter_x + box_size, y + jitter_y + box_size),
                      color, thickness=-1)
        # overlay text
        cv2.putText(frame, mode, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, mode, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(frame)

    vw.release()

def write_labels(labels, out_dir):
    with open(os.path.join(out_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for c in labels:
            f.write(c + "\n")

def main():
    ap = argparse.ArgumentParser(description="Generate a tiny synthetic gesture dataset.")
    ap.add_argument("--out", default="data", help="Output directory (default: data)")
    ap.add_argument("--classes", nargs="+",
                    default=["swipe_left", "swipe_right", "stop"],
                    help="Class names (default: swipe_left swipe_right stop)")
    ap.add_argument("--clips", type=int, default=16, help="Clips per class (default: 16)")
    ap.add_argument("--seconds", type=float, default=1.5, help="Seconds per clip (default: 1.5)")
    ap.add_argument("--fps", type=int, default=16, help="Frames per second (default: 16)")
    ap.add_argument("--size", type=int, default=224, help="Frame size WxH (default: 224)")
    ap.add_argument("--box", type=int, default=60, help="Box size (default: 60)")
    ap.add_argument("--codec", default="mp4v", help="Codec fourcc (mp4v or XVID)")
    ap.add_argument("--ext", default=".mp4", help="File extension (.mp4 or .avi)")
    args = ap.parse_args()

    ensure_dir(args.out)
    write_labels(args.classes, ".")  # writes labels.txt to project root

    print(f"Generating synthetic dataset -> {args.out}")
    for cls in args.classes:
        cls_dir = os.path.join(args.out, cls)
        ensure_dir(cls_dir)
        mode = "stop" if cls == "stop" else ("swipe_left" if "left" in cls else ("swipe_right" if "right" in cls else "stop"))
        for i in range(args.clips):
            filename = os.path.join(cls_dir, f"{cls}_{i+1:03d}{args.ext}")
            make_clip(
                mode=mode,
                out_path=filename,
                seconds=args.seconds,
                fps=args.fps,
                size=args.size,
                box_size=args.box,
                seed=i + 1,
                codec=args.codec
            )
        print(f"  {cls}: {args.clips} clips")

    print("Done. You can now run: python train.py, python export_onnx.py, python app.py")

if __name__ == "__main__":
    main()
