"""
app.py

A minimal REST API + browser UI that serves the trained model.

Usage (local — defaults are enough):
    python app.py

Optional overrides:
    python app.py --checkpoint checkpoints\\best.pt --port 5000

Environment variables (useful on Hugging Face Spaces / cloud):
    CHECKPOINT   path to .pt weights  (default: checkpoints/best.pt)
    HOST         bind address         (default: 0.0.0.0)
    PORT         bind port            (default: 5000)

Open the browser UI at http://localhost:5000/

Or send a page/line image via the API:

    curl -X POST http://localhost:5000/recognize_page ^
        -F "image=@path\\to\\page.jpg"

    curl -X POST http://localhost:5000/recognize_line ^
        -F "image=@path\\to\\line.jpg"

Response (JSON):
    { "lines": ["...", "..."], "full_text": "...\\n...", "line_count": N }
    { "text": "..." }
"""

import argparse
import os
import tempfile

import cv2
import torch
from flask import Flask, request, jsonify, render_template

from dataset import preprocess_image
from model import CRNN
from segment_lines import segment_page
from infer import ctc_greedy_decode_single

# Project root = folder containing this file (works locally and on HF Spaces)
_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "best.pt")
_FALLBACK_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "last.pt")

app = Flask(__name__)

# populated at startup in main()
_model = None
_idx2char = None
_device = None


@app.after_request
def add_cors_headers(response):
    # Lets a separately hosted UI (HF Space, static site, etc.) call this API.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    # Avoid browsers keeping a stale UI that calls the wrong API style
    if request.path == "/" or response.content_type and "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


def recognize_line_image(img_path):
    img = preprocess_image(img_path, augment=False)
    tensor = torch.from_numpy(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)[0]
    return ctc_greedy_decode_single(logits, _idx2char)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", ui_version="2026-08-30-gradio")


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return ("", 204)
    return jsonify({"status": "ok", "device": str(_device)})


@app.route("/recognize_line", methods=["POST", "OPTIONS"])
def recognize_line():
    if request.method == "OPTIONS":
        return ("", 204)
    if "image" not in request.files:
        return jsonify({"error": "no 'image' file in request"}), 400

    file = request.files["image"]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        text = recognize_line_image(tmp_path)
        return jsonify({"text": text})
    finally:
        os.unlink(tmp_path)


@app.route("/recognize_page", methods=["POST", "OPTIONS"])
def recognize_page():
    if request.method == "OPTIONS":
        return ("", 204)
    if "image" not in request.files:
        return jsonify({"error": "no 'image' file in request"}), 400

    file = request.files["image"]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        _, line_imgs = segment_page(tmp_path)
        results = []
        with tempfile.TemporaryDirectory() as line_dir:
            for i, line_img in enumerate(line_imgs, 1):
                line_path = os.path.join(line_dir, f"line_{i}.jpg")
                cv2.imwrite(line_path, line_img)
                results.append(recognize_line_image(line_path))

        return jsonify({
            "lines": results,
            "full_text": "\n".join(results),
            "line_count": len(results),
        })
    finally:
        os.unlink(tmp_path)


def resolve_checkpoint(cli_value):
    """Pick checkpoint from CLI, env, or default files next to app.py."""
    candidates = []
    if cli_value:
        candidates.append(cli_value)
    env = os.environ.get("CHECKPOINT")
    if env:
        candidates.append(env)
    candidates.extend([_DEFAULT_CHECKPOINT, _FALLBACK_CHECKPOINT])

    for path in candidates:
        options = [path]
        if not os.path.isabs(path):
            options.append(os.path.join(_ROOT, path))
        for opt in options:
            if opt and os.path.isfile(opt):
                return os.path.abspath(opt)

    tried = ", ".join(dict.fromkeys(candidates))
    raise FileNotFoundError(
        f"No checkpoint found. Looked for: {tried}. "
        "Place best.pt under checkpoints/ or set CHECKPOINT / --checkpoint."
    )


def main():
    global _model, _idx2char, _device

    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", "5000"))

    ap = argparse.ArgumentParser(description="BN-HTR API + UI")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Path to .pt weights (default: checkpoints/best.pt or $CHECKPOINT)",
    )
    ap.add_argument("--host", default=default_host)
    ap.add_argument("--port", type=int, default=default_port)
    args = ap.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint, map_location=_device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=_device)
    _idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    _model = CRNN(num_classes=num_classes).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    print(f"Checkpoint: {checkpoint}")
    print(f"Model loaded on {_device}. Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
