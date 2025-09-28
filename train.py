# train.py (robust + verbose, Windows/CPU friendly)
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # harmless if TF not installed

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import timm
from dataset import GestureClips, read_labels

# keep things reproducible
torch.manual_seed(42)

class ViTTemporal(nn.Module):
    """Frame-wise ViT encoder -> mean pool over time -> linear head."""
    def __init__(self, num_classes, vit_name="vit_tiny_patch16_224"):
        super().__init__()
        print(f"[init] creating backbone: {vit_name}", flush=True)
        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = self.vit.num_features
        self.head = nn.Linear(feat_dim, num_classes)
        print(f"[init] backbone ready, feat_dim={feat_dim}", flush=True)

    def forward(self, x):  # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.vit(x)                        # (B*T, D)
        feats = feats.view(B, T, -1).mean(dim=1)   # (B, D)
        return self.head(feats)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}", flush=True)

    labels, _ = read_labels("labels.txt")
    n_classes = len(labels)
    print(f"[info] labels={labels}", flush=True)
    if n_classes < 2:
        raise ValueError("labels.txt must contain at least two classes (one per line).")

    print("[info] building datasets...", flush=True)
    train_ds = GestureClips(train=True)
    val_ds   = GestureClips(train=False)
    print(f"[info] Train clips: {len(train_ds)} | Val clips: {len(val_ds)}", flush=True)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("No data in train/val. Check data/<class>/*.mp4 and labels.txt.")

    # CPU/Windows-friendly loaders
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=0, pin_memory=False)

    print("[info] building model...", flush=True)
    model = ViTTemporal(num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # ---- Smoke test: fetch ONE batch and do ONE forward/backward ----
    print("[smoke] fetching one batch...", flush=True)
    xb, yb = next(iter(train_dl))
    print(f"[smoke] batch shapes: x={tuple(xb.shape)} y={tuple(yb.shape)}", flush=True)
    xb, yb = xb.to(device), yb.to(device)
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()   # prove backward works
    optimizer.zero_grad(set_to_none=True)
    print("[smoke] forward/backward OK", flush=True)

    # ---- Real training ----
    best_acc = 0.0
    epochs = 3
    print("[train] starting training loop...", flush=True)
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for step, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            if step % 10 == 0:
                print(f"[train] epoch {epoch:02d} step {step} loss {loss.item():.4f}", flush=True)

        scheduler.step()
        train_acc = correct / total if total else 0.0
        train_loss = loss_sum / total if total else 0.0

        # ---- Validate ----
        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vcorrect += (logits.argmax(1) == y).sum().item()
                vtotal += x.size(0)
        val_acc = vcorrect / vtotal if vtotal else 0.0

        print(f"[epoch {epoch:02d}] train_loss {train_loss:.4f} | train_acc {train_acc:.3f} | "
              f"val_acc {val_acc:.3f} | time {(time.time()-t_ep):.1f}s", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "vit_temporal_best.pt")
            print(f"[save] best so far: val_acc={best_acc:.3f} -> vit_temporal_best.pt", flush=True)

    print(f"[done] best val acc: {best_acc:.3f}", flush=True)

if __name__ == "__main__":
    # Show exceptions if the interpreter dies abruptly
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", flush=True)
        raise
