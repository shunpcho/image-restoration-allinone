"""Inference / evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from image_restoration_allinone.configs.config import LossConfig, ModelConfig
from image_restoration_allinone.data.dataset import PairedRestorationDataset
from image_restoration_allinone.data.transforms import build_val_transform
from image_restoration_allinone.models.build import build_model
from image_restoration_allinone.utils.evaluator import Evaluator
from image_restoration_allinone.utils.loss import LossComposer
from image_restoration_allinone.utils.visualizer import save_comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference / evaluation with a trained restoration model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate.")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results"))
    parser.add_argument("--arch-name", type=str, default="NAFNet", help="Model architecture name.")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--save-images", action="store_true", help="Save side-by-side comparison images.")
    return parser.parse_args()


def _save_images(
    model: torch.nn.Module,
    val_loader: DataLoader[dict[str, torch.Tensor]],
    output_dir: Path,
    device: torch.device,
) -> None:
    """Run inference and write side-by-side comparison images to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for idx, batch in enumerate(val_loader):
            degraded: torch.Tensor = batch["degraded"].to(device)
            clean: torch.Tensor = batch["clean"].to(device)
            restored = model(degraded)
            save_comparison(
                degraded[0],
                restored[0],
                clean[0],
                output_dir / f"{idx:04d}.png",
            )
    print(f"Saved comparison images to {output_dir}")


def main() -> None:
    """Load a checkpoint and evaluate on the specified split."""
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model_cfg = ModelConfig(arch_name=args.arch_name, width=args.width)
    model = build_model(model_cfg).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint} (iter {ckpt.get('iteration', '?')})")

    # Data
    val_transform = build_val_transform()
    dataset = PairedRestorationDataset(args.data_root, split=args.split, transform=val_transform)
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(dataset, batch_size=1, shuffle=False)

    # Loss (MSE default for evaluation)
    criterion = LossComposer(LossConfig())

    # Evaluate
    evaluator = Evaluator(model, criterion, val_loader, device)
    metrics = evaluator.run()

    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    final_score = metrics["val/psnr_y"] + 10.0 * metrics["val/ssim_y"] - 5.0 * metrics["val/lpips"]
    print(f"\n  Final_Score (PSNR_Y + 10*SSIM_Y - 5*LPIPS): {final_score:.6f}")

    # Optional: save comparison images
    if args.save_images:
        _save_images(model, val_loader, args.output_dir, device)


if __name__ == "__main__":
    main()
