# app.py — robust across Gradio versions + auto-detect ONNX IO names
import os, tempfile, cv2, torch, onnxruntime, numpy as np
import gradio as gr
from dataset import read_labels

T = 16
SIZE = 224
MODEL_PATH = "vit_temporal.onnx"

labels, _ = read_labels("labels.txt")

# --- ONNX session + auto-detect names ---
ort_session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
# detect first input and first output names to avoid mismatches
INPUT_NAME = ort_session.get_inputs()[0].name   # e.g. "input" or "video"
OUTPUT_NAME = ort_session.get_outputs()[0].name # e.g. "logits" or something else

def preprocess_clip(frames_rgb):
    if len(frames_rgb) == 0:
        frames_rgb = [np.zeros((SIZE, SIZE, 3), dtype=np.uint8)]
    if len(frames_rgb) < T:
        frames_rgb = frames_rgb + [frames_rgb[-1]] * (T - len(frames_rgb))
    frames_rgb = frames_rgb[:T]
    clip = [cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA) for f in frames_rgb]
    clip = np.stack(clip, axis=0)                                    # (T,H,W,3)
    clip = np.transpose(clip, (0, 3, 1, 2)).astype(np.float32) / 255 # (T,3,H,W)
    clip = (clip - 0.5) / 0.5
    clip = np.expand_dims(clip, 0)                                   # (1,T,3,H,W)
    return clip

def _extract_path_from_gradio_video(inp):
    if isinstance(inp, str) and os.path.exists(inp):
        return inp
    if isinstance(inp, dict):
        for key in ("video", "name", "path", "filepath"):
            v = inp.get(key)
            if isinstance(v, str) and os.path.exists(v):
                return v
        for key in ("data", "video"):
            v = inp.get(key)
            if isinstance(v, (bytes, bytearray)):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(v); tmp.flush(); tmp.close()
                return tmp.name
    if isinstance(inp, (list, tuple)) and inp and isinstance(inp[0], str) and os.path.exists(inp[0]):
        return inp[0]
    return None

def _read_uniform_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total - 1, max(T, 1)).astype(int)
    want = set(int(i) for i in idxs.tolist())
    j = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if j in want:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        j += 1
    cap.release()
    return frames

def predict_from_video(gradio_video):
    video_path = _extract_path_from_gradio_video(gradio_video)
    if not video_path or not os.path.exists(video_path):
        return {}
    frames = _read_uniform_frames(video_path)

    # If OpenCV choked on the codec (common with recorded webm), re-encode once:
    if len(frames) == 0:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); tmp_name = tmp.name; tmp.close()
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        out = cv2.VideoWriter(tmp_name, fourcc, 20.0, (w, h))
        while True:
            ok, frame = cap.read()
            if not ok: break
            out.write(frame)
        cap.release(); out.release()
        frames = _read_uniform_frames(tmp_name)

    clip = preprocess_clip(frames)
    # >>> use the detected ONNX input/output names <<<
    logits = ort_session.run([OUTPUT_NAME], {INPUT_NAME: clip})[0]  # (1, C)
    probs = torch.softmax(torch.from_numpy(logits), dim=1)[0].numpy().tolist()
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

def predict_from_image(image):
    if image is None:
        return {}
    clip = preprocess_clip([image] * T)
    logits = ort_session.run([OUTPUT_NAME], {INPUT_NAME: clip})[0]
    probs = torch.softmax(torch.from_numpy(logits), dim=1)[0].numpy().tolist()
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

with gr.Blocks() as demo:
    gr.Markdown("# Gesture Classifier (ONNX)\nRecord or upload a short video, then click **Classify Video**.")
    with gr.Tab("Video (record or upload)"):
        vid_in = gr.Video(label="Record from webcam or upload a short clip")
        vid_out = gr.Label(num_top_classes=3, label="Prediction")
        gr.Button("Classify Video").click(fn=predict_from_video, inputs=vid_in, outputs=vid_out)
    with gr.Tab("Single Image (fallback)"):
        img_in = gr.Image(label="Upload an image frame", type="numpy")
        img_out = gr.Label(num_top_classes=3, label="Prediction")
        gr.Button("Classify Image").click(fn=predict_from_image, inputs=img_in, outputs=img_out)

if __name__ == "__main__":
    demo.launch()
