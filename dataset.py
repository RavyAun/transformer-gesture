import os, glob, random, cv2, numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T

def read_labels(path="labels.txt"):
    with open(path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    if len(labels) < 2:
        raise ValueError("labels.txt must list at least two classes (one per line).")
    idx = {c: i for i, c in enumerate(labels)}
    return labels, idx

def sample_clip_frames(video_path, num_frames=16, resize=224):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, max(0, total - 1), num_frames).astype(int)

    j = 0
    want = set(int(i) for i in idxs.tolist())
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if j in want:
            # BGR -> RGB, resize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        j += 1
    cap.release()

    # If video is short, pad last frame
    if len(frames) == 0:
        # create a blank frame to avoid crashing
        frames = [np.zeros((resize, resize, 3), dtype=np.uint8)]
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # (T, H, W, 3) uint8
    return np.stack(frames, axis=0).astype(np.uint8)

class GestureClips(Dataset):
    def __init__(self, root="data", labels_path="labels.txt",
                 num_frames=16, resize=224, train=True):
        self.labels, self.label_to_idx = read_labels(labels_path)
        self.items = []
        for cls in self.labels:
            folder = os.path.join(root, cls)
            for p in glob.glob(os.path.join(folder, "*")):
                if p.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                    self.items.append((p, self.label_to_idx[cls]))

        if len(self.items) == 0:
            raise ValueError(
                f"No videos found under '{root}/<class>/*.mp4'. "
                f"Create folders for your classes ({', '.join(self.labels)}) and add clips."
            )

        random.seed(42)
        random.shuffle(self.items)
        split = int(0.8 * len(self.items)) if len(self.items) > 1 else len(self.items)
        self.items = self.items[:split] if train else self.items[split:]
        self.num_frames = num_frames
        self.resize = resize
        self.train = train

        # Apply ToTensor (HWC->CHW) + Normalize per-frame
        self.tx = T.Compose([
            T.ToTensor(),                             # -> float [0,1], CHW
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])          # -> roughly [-1,1]
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        clip = sample_clip_frames(path, self.num_frames, self.resize)  # (T, H, W, 3), uint8

        # Simple augmentation: horizontal flip (prob 0.5)
        if self.train and random.random() < 0.5:
            # Use np.flip(...).copy() to avoid negative strides (Windows-safe)
            clip = np.flip(clip, axis=2).copy()

        # Per-frame transforms on HWC -> CHW tensors
        frames_t = [self.tx(frame) for frame in clip]   # list of (C,H,W) torch tensors
        clip_t = torch.stack(frames_t, dim=0)           # (T, C, H, W)

        return clip_t.float(), torch.tensor(y, dtype=torch.long)
