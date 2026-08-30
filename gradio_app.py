"""
gradio_app.py

Gradio interface for BN-HTR (Hugging Face Spaces).

On ZeroGPU Spaces, decorate inference with @spaces.GPU.
Upload this file as app.py in the Space repo.
"""

# On ZeroGPU, import spaces before torch. Locally, spaces may be missing.
try:
    import spaces
except ImportError:  # local / non-HF
    class spaces:  # type: ignore
        @staticmethod
        def GPU(duration=60):
            def decorator(fn):
                return fn
            return decorator

import os

import gradio as gr
import torch

from model import CRNN
from recognize import Recognizer
from segment_lines import resolve_detector_weights

CHECKPOINT_PATH = "checkpoints/best.pt"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_recognizer = None


def load_model():
    global _recognizer, _device
    if _recognizer is not None:
        return

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}.")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=_device, weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=_device)

    idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1
    model = CRNN(num_classes=num_classes).to(_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _recognizer = Recognizer(model, idx2char, _device)


@spaces.GPU(duration=120)
def recognize(image_path):
    if image_path is None:
        return "Please upload an image."
    load_model()
    result = _recognizer.recognize(image_path)
    if not result["full_text"].strip():
        return "No text detected."
    return result["full_text"]


@spaces.GPU(duration=60)
def recognize_line(image_path):
    if image_path is None:
        return "Please upload an image."
    load_model()
    return _recognizer.recognize(image_path, force_mode="line")["full_text"]


@spaces.GPU(duration=120)
def recognize_page(image_path):
    if image_path is None:
        return "Please upload an image."
    load_model()
    return _recognizer.recognize(image_path, force_mode="page")["full_text"]


with gr.Blocks(title="Bangla Handwritten Text Recognition") as demo:
    detector = resolve_detector_weights()
    gr.Markdown(
        "# Bangla Handwritten Text Recognition (BN-HTR)\n"
        "Upload a handwritten Bangla page or line image. "
        "Lines are detected automatically and recognized with CRNN+CTC."
        + (
            "\n\nLine detector: YOLO."
            if detector
            else "\n\nLine detector: classical CV "
            "(add `checkpoints/line_detector.pt` for YOLO)."
        )
    )

    with gr.Tab("Recognize"):
        auto_input = gr.Image(type="filepath", label="Handwritten image")
        auto_button = gr.Button("Recognize", variant="primary")
        auto_output = gr.Textbox(label="Recognized text", lines=12)
        auto_button.click(
            fn=recognize, inputs=auto_input, outputs=auto_output, api_name="recognize"
        )

    with gr.Tab("Single line"):
        line_input = gr.Image(type="filepath", label="Cropped line image")
        line_button = gr.Button("Recognize")
        line_output = gr.Textbox(label="Recognized text")
        line_button.click(
            fn=recognize_line,
            inputs=line_input,
            outputs=line_output,
            api_name="recognize_line",
        )

    with gr.Tab("Full page"):
        page_input = gr.Image(type="filepath", label="Full handwritten page")
        page_button = gr.Button("Recognize")
        page_output = gr.Textbox(label="Recognized text", lines=15)
        page_button.click(
            fn=recognize_page,
            inputs=page_input,
            outputs=page_output,
            api_name="recognize_page",
        )

if __name__ == "__main__":
    demo.launch()
