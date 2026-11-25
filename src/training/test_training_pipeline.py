import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.ucf101_dataset import UCF101Dataset
from src.models.two_stream_model import TwoStreamActionRecognition


def main():
    parser = argparse.ArgumentParser(
        description="Run one forward/backward/optimizer step on a real UCF101 batch."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Directory containing the 51 UCF101_processed ZIP archives.",
    )
    parser.add_argument(
        "--split-dir",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data" / "splits" / "ucf101" / "split_1"
        ),
        help="Directory containing the generated split files and class mapping.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for this small smoke test, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Initialize the RGB/flow ResNet-18 backbones from ImageNet weights.",
    )
    args = parser.parse_args()
    torch.manual_seed(42)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    root_dir = args.data_root
    split_dir = Path(args.split_dir)

    dataset = UCF101Dataset(
        root_dir=root_dir,
        split_file=split_dir / "train.txt",
        class_mapping_file=split_dir / "class_mapping.csv",
        num_frames=32,
        img_size=224,
    )

    assert len(dataset) == 8492
    assert len(dataset.classes) == 100

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(loader))

    rgb = batch["rgb"].to(device)
    flow = batch["flow"].to(device)
    labels = batch["label"].to(device)

    model = TwoStreamActionRecognition(
        num_classes=100,
        num_frames=32,
        frame_batch_size=4,
        hidden_dim=256,
        dropout=0.5,
        pretrained=args.pretrained,
    ).to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    # Save one parameter before the optimizer update.
    parameter_before = next(model.parameters()).detach().clone()

    optimizer.zero_grad(set_to_none=True)

    outputs = model(rgb, flow)
    loss = criterion(outputs, labels)

    assert torch.isfinite(loss), "Loss is NaN or Inf"

    loss.backward()

    gradients = [
        p.grad
        for p in model.parameters()
        if p.grad is not None
    ]

    assert len(gradients) > 0, "No gradients were calculated"
    assert all(torch.isfinite(g).all() for g in gradients), (
        "Some gradients contain NaN or Inf"
    )

    gradient_count = len(gradients)

    optimizer.step()

    parameter_after = next(model.parameters()).detach()
    parameter_changed = not torch.equal(
        parameter_before,
        parameter_after,
    )

    print("\\n========== FULL TRAINING PIPELINE TEST ==========")
    print("Dataset videos:", len(dataset))
    print("Dataset classes:", len(dataset.classes))
    print("RGB input:", rgb.shape)
    print("Flow input:", flow.shape)
    print("Labels:", labels.tolist())
    print("Model output:", outputs.shape)
    print("Loss:", loss.item())
    print("Parameters with gradients:", gradient_count)
    print("All gradients finite: PASSED")
    print("Optimizer step: PASSED")
    print("Model parameter changed:", parameter_changed)

    assert outputs.shape == (2, 100)
    assert parameter_changed, "Model parameters did not change"

    print("\\nFULL TRAINING PIPELINE TEST PASSED!")


if __name__ == "__main__":
    main()
