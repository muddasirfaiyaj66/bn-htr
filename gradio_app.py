"""
gradio_app.py

Gradio interface for the Bangla HTR model, built to deploy directly on
Hugging Face Spaces (free CPU tier). Provides both a web UI and an
automatically generated API endpoint (Gradio exposes every Space as an API
that can be called with `gradio_client` or plain HTTP POST — see the
"Use via API" link at the bottom of any deployed Space's page).

--------------------------------------------------------------------------
DEPLOYING THIS ON HUGGING FACE SPACES
--------------------------------------------------------------------------
1. Create a Space at https://huggingface.co/new-space
     - SDK: Gradio
     - Hardware: CPU basic (free)

2. In the Space's repo, upload these files from this project:
     - dataset.py
     - model.py
     - segment_lines.py
     - infer.py
     - requirements_hf.txt   -> rename to requirements.txt in the Space repo
     - this file             -> rename to app.py in the Space repo
       (Spaces auto-runs whatever file is named app.py)

3. Upload your trained checkpoint (checkpoints/best.pt) into the Space repo
   too, at the path "checkpoints/best.pt" (matching CHECKPOINT_PATH below).
   The HF web uploader automatically uses Git LFS for files over ~10MB, so
   just drag-and-drop it in — no special steps needed.

4. The Space builds automatically. Once it's live, you'll have:
     - a web UI at https://huggingface.co/spaces/<your-username>/<space-name>
     - a free API automatically, callable like:

         from gradio_client import Client
         client = Client("<your-username>/<space-name>")
         result = client.predict("path/to/line.jpg", api_name="/recognize_line")
         print(result)

     or via plain HTTP (see the "Use via API" button on the Space page for
     the exact request format once it's deployed).
--------------------------------------------------------------------------
"""

import os
import tempfile

import cv2
import gradio as gr
import torch

from dataset import preprocess_image
from model import CRNN
from segment_lines import segment_page
from infer import ctc_greedy_decode_single

CHECKPOINT_PATH = "checkpoints/best.pt"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_idx2char = None


def load_model():
    global _model, _idx2char
    if _model is not None:
        return

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. Upload your trained "
            f"best.pt to this exact path in the Space repo."
        )

    ckpt = torch.load(CHECKPOINT_PATH, map_location=_device)
    _idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    _model = CRNN(num_classes=num_classes).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()


def _recognize_line_path(img_path):
    load_model()
    img = preprocess_image(img_path, augment=False)
    tensor = torch.from_numpy(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)[0]
    return ctc_greedy_decode_single(logits, _idx2char)


def recognize_line(image_path):
    """Recognize text from a single pre-cropped line image."""
    if image_path is None:
        return "Please upload an image."
    return _recognize_line_path(image_path)


def recognize_page(image_path):
    """Segment a full handwritten page into lines and recognize each one."""
    if image_path is None:
        return "Please upload an image."

    load_model()
    _, line_imgs = segment_page(image_path)

    if len(line_imgs) == 0:
        return "No lines were detected in this image."

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, line_img in enumerate(line_imgs, 1):
            line_path = os.path.join(tmp, f"line_{i}.jpg")
            cv2.imwrite(line_path, line_img)
            text = _recognize_line_path(line_path)
            results.append(text)

    return "\n".join(results)


with gr.Blocks(title="Bangla Handwritten Text Recognition") as demo:
    gr.Markdown(
        "# Bangla Handwritten Text Recognition (BN-HTR)\n"
        "CRNN + CTC model trained on the BN-HTRd dataset. "
        "Upload a single cropped handwritten line, or a full handwritten page."
    )

    with gr.Tab("Recognize a single line"):
        line_input = gr.Image(type="filepath", label="Cropped line image")
        line_button = gr.Button("Recognize")
        line_output = gr.Textbox(label="Recognized text")
        line_button.click(fn=recognize_line, inputs=line_input, outputs=line_output,
                           api_name="recognize_line")

    with gr.Tab("Recognize a full page"):
        page_input = gr.Image(type="filepath", label="Full handwritten page image")
        page_button = gr.Button("Recognize")
        page_output = gr.Textbox(label="Recognized text (one line per row)", lines=15)
        page_button.click(fn=recognize_page, inputs=page_input, outputs=page_output,
                           api_name="recognize_page")

if __name__ == "__main__":
    demo.launch()
