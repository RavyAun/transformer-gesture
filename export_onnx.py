import torch
from train import ViTTemporal
from dataset import read_labels

def export_onnx():
    labels, _ = read_labels("labels.txt")
    n = len(labels)
    if n < 2:
        raise ValueError("labels.txt must contain at least two classes.")
    model = ViTTemporal(num_classes=n)
    model.load_state_dict(torch.load("vit_temporal_best.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 16, 3, 224, 224)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model, dummy, "vit_temporal.onnx",
        input_names=["input"], output_names=["logits"],
        dynamic_axes=dynamic_axes, opset_version=17
    )
    print("Exported to vit_temporal.onnx")

if __name__ == "__main__":
    export_onnx()
