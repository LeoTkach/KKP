"""Gradio demo for AI-generated image detection."""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

import gradio as gr
import torch
from dotenv import load_dotenv
from PIL import Image

from kkp.config import load_config
from kkp.data.transforms import get_transforms
from kkp.inference import predict_image
from kkp.training import load_checkpoint

logger = logging.getLogger(__name__)

LABELS_UA = {
    "real": "Реальне фото",
    "ai_generated": "AI-генерація",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KKP Gradio demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ai_generated.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    return parser.parse_args()


def build_predict_fn(
    config_path: Path,
    checkpoint_path: Path | None,
):
    config = load_config(config_path)
    training = config["training"]
    output_dir = Path(config["paths"]["output_dir"])
    resolved_checkpoint = (
        checkpoint_path if checkpoint_path else output_dir / "checkpoints" / "best.pth"
    )

    if not resolved_checkpoint.is_file():
        msg = f"Checkpoint not found: {resolved_checkpoint}"
        raise FileNotFoundError(msg)

    device = torch.device(training["device"])
    model, checkpoint = load_checkpoint(resolved_checkpoint, device)
    transform = get_transforms(training["image_size"], train=False)
    class_names = tuple(config["data"]["classes"])

    logger.info(
        "Loaded %s (epoch %d) from %s",
        config["model"]["name"],
        checkpoint["epoch"],
        resolved_checkpoint,
    )

    def predict(image: Image.Image) -> tuple[str, dict[str, float]]:
        if image is None:
            return "Завантажте зображення", {}

        label, confidences = predict_image(model, image, transform, device, class_names)
        result = LABELS_UA.get(label, label)
        display = {LABELS_UA.get(name, name): score for name, score in confidences.items()}
        return result, display

    return predict


def create_demo(predict_fn) -> gr.Blocks:
    with gr.Blocks(title="KKP — AI vs Real") as demo:
        gr.Markdown(
            """
            # KKP — детекція AI-зображень
            Завантажте фото, щоб перевірити, чи воно **реальне** чи **AI-генероване**.
            """,
        )
        with gr.Row():
            image_input = gr.Image(type="pil", label="Зображення")
            with gr.Column():
                label_output = gr.Label(label="Ймовірності")
                text_output = gr.Textbox(label="Результат", interactive=False)

        image_input.change(predict_fn, inputs=image_input, outputs=[text_output, label_output])

    return demo


def _resolve_server_port(host: str, preferred: int, *, attempts: int = 10) -> int:
    for offset in range(attempts):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    msg = f"No free port in range {preferred}-{preferred + attempts - 1}"
    raise OSError(msg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()

    predict_fn = build_predict_fn(args.config, args.checkpoint)
    demo = create_demo(predict_fn)
    port = _resolve_server_port(args.host, args.port)
    if port != args.port:
        logger.info("Port %d busy, using %d instead", args.port, port)
    demo.launch(server_name=args.host, server_port=port, share=args.share)


if __name__ == "__main__":
    main()
