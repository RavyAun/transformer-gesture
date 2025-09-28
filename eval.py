# eval.py
import torch
from dataset import GestureClips, read_labels
from train import ViTTemporal

labels, _ = read_labels("labels.txt")
n_classes = len(labels)

# Load validation data
val_ds = GestureClips(train=False)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)

# Load trained model
model = ViTTemporal(num_classes=n_classes)
model.load_state_dict(torch.load("vit_temporal_best.pt", map_location="cpu"))
model.eval()

correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in val_dl:
        logits = model(x)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())

print(f"Validation accuracy: {correct/total:.2%}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

