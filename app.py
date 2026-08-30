"""
app.py

Flask REST API and browser UI for Bangla HTR.

Usage:
    python app.py
    python app.py --checkpoint checkpoints\\best.pt --port 5000

Environment:
    CHECKPOINT, LINE_DETECTOR, HOST, PORT

Endpoints:
    POST /recognize
    POST /recognize_page
    POST /recognize_line
    GET  /health
"""

import argparse
import os
import tempfile

from flask import Flask, request, jsonify, render_template
import torch

from model import CRNN
from recognize import Recognizer
from segment_lines import resolve_detector_weights

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "best.pt")
_FALLBACK_CHECKPOINT = os.path.join(_ROOT, "checkpoints", "last.pt")

app = Flask(__name__)

_recognizer = None
_device = None


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    if request.path == "/" or (
        response.content_type and "text/html" in response.content_type
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _save_upload():
    if "image" not in request.files:
        return None, (jsonify({"error": "no 'image' file in request"}), 400)
    file = request.files["image"]
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    file.save(tmp.name)
    tmp.close()
    return tmp.name, None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", ui_version="2026-08-31-nav2")


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return ("", 204)
    detector = resolve_detector_weights()
    return jsonify(
        {
            "status": "ok",
            "device": str(_device),
            "line_detector": bool(detector),
            "line_detector_path": detector,
        }
    )


@app.route("/recognize", methods=["POST", "OPTIONS"])
def recognize():
    if request.method == "OPTIONS":
        return ("", 204)
    tmp_path, err = _save_upload()
    if err:
        return err
    try:
        result = _recognizer.recognize(tmp_path)
        return jsonify(result)
    finally:
        os.unlink(tmp_path)


@app.route("/recognize_line", methods=["POST", "OPTIONS"])
def recognize_line():
    if request.method == "OPTIONS":
        return ("", 204)
    tmp_path, err = _save_upload()
    if err:
        return err
    try:
        result = _recognizer.recognize(tmp_path, force_mode="line")
        return jsonify({"text": result["full_text"]})
    finally:
        os.unlink(tmp_path)


@app.route("/recognize_page", methods=["POST", "OPTIONS"])
def recognize_page():
    if request.method == "OPTIONS":
        return ("", 204)
    tmp_path, err = _save_upload()
    if err:
        return err
    try:
        result = _recognizer.recognize(tmp_path, force_mode="page")
        return jsonify(result)
    finally:
        os.unlink(tmp_path)


def resolve_checkpoint(cli_value):
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
    global _recognizer, _device

    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", "5000"))

    ap = argparse.ArgumentParser(description="BN-HTR API + UI")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default=default_host)
    ap.add_argument("--port", type=int, default=default_port)
    args = ap.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(checkpoint, map_location=_device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=_device)
    idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    model = CRNN(num_classes=num_classes).to(_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _recognizer = Recognizer(model, idx2char, _device)

    detector = resolve_detector_weights()
    print(f"Checkpoint: {checkpoint}")
    print(f"Line detector: {detector or '(classical fallback)'}")
    print(f"Model loaded on {_device}. Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
