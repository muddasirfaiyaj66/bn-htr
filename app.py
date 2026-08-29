"""
app.py

A minimal REST API that serves the trained model, so it can be called from
any other application (a web frontend, mobile app, another backend service,
a Postman/curl test, etc.) instead of running it purely from the command line.

Usage:
    python app.py --checkpoint checkpoints\\best.pt --port 5000

Open the browser UI at http://localhost:5000/

Or send a page/line image via the API:

    curl -X POST http://localhost:5000/recognize_page ^
        -F "image=@path\\to\\page.jpg"

    curl -X POST http://localhost:5000/recognize_line ^
        -F "image=@path\\to\\line.jpg"

Response (JSON):
    { "lines": ["...", "..."], "full_text": "...\\n..." }        (recognize_page)
    { "text": "..." }                                             (recognize_line)
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
    return response


def recognize_line_image(img_path):
    img = preprocess_image(img_path, augment=False)
    tensor = torch.from_numpy(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)[0]
    return ctc_greedy_decode_single(logits, _idx2char)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


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


def main():
    global _model, _idx2char, _device

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=_device)
    _idx2char = ckpt["idx2char"]
    num_classes = len(ckpt["char2idx"]) + 1

    _model = CRNN(num_classes=num_classes).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    print(f"Model loaded on {_device}. Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
