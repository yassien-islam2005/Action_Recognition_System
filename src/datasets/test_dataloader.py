import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.ucf101_dataset import UCF101Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Check the UCF101 RGB/optical-flow dataloaders."
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
        help="Directory containing train.txt, val.txt, test.txt and class_mapping.csv.",
    )
    args = parser.parse_args()
    root_dir = args.data_root
    split_dir = Path(args.split_dir)

    expected_counts = {
        "train": 8492,
        "val": 941,
        "test": 3742,
    }

    print("\n========== THREE SPLITS DATALOADER TEST ==========")

    for split_name, expected_count in expected_counts.items():
        print(f"\n===== {split_name.upper()} =====")

        dataset = UCF101Dataset(
            root_dir=root_dir,
            split_file=split_dir / f"{split_name}.txt",
            class_mapping_file=split_dir / "class_mapping.csv",
            num_frames=32,
            img_size=224,
        )

        assert len(dataset) == expected_count, (
            f"{split_name}: expected {expected_count}, "
            f"got {len(dataset)}"
        )

        assert len(dataset.classes) == 100
        assert "Archery" not in dataset.classes

        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        batch = next(iter(loader))

        rgb = batch["rgb"]
        flow = batch["flow"]
        labels = batch["label"]

        print("Dataset videos:", len(dataset))
        print("Classes:", len(dataset.classes))
        print("RGB Batch:", rgb.shape)
        print("Flow Batch:", flow.shape)
        print("Labels:", labels.shape)
        print("Label range:", labels.min().item(), "to", labels.max().item())
        print("Video names:", batch["video_name"])
        print("Class names:", batch["class_name"])

        assert rgb.shape == (2, 32, 3, 224, 224)
        assert flow.shape == (2, 32, 2, 224, 224)
        assert labels.shape == (2,)

        assert torch.isfinite(rgb).all()
        assert torch.isfinite(flow).all()

        assert labels.min().item() >= 0
        assert labels.max().item() < 100

        for class_name, label in zip(
            batch["class_name"], labels.tolist()
        ):
            assert dataset.class_to_idx[class_name] == label

        print(f"{split_name.upper()} TEST: PASSED")

    print("\nALL THREE DATALOADER TESTS PASSED!")


if __name__ == "__main__":
    main()
