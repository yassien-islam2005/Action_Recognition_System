import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UCF101Dataset(Dataset):
    """
    UCF101 Two-Stream Dataset.

    Reads only videos listed in the selected split file.
    Uses a shared class_mapping.csv for consistent labels.
    Reads RGB and DIS Optical Flow directly from ZIP archives.
    """

    def __init__(
        self,
        root_dir,
        split_file,
        class_mapping_file=None,
        num_frames=32,
        img_size=224,
    ):
        self.root_dir = Path(root_dir)
        self.split_file = Path(split_file)
        self.num_frames = num_frames
        self.img_size = img_size

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        if not self.split_file.is_file():
            raise FileNotFoundError(
                f"Split file not found: {self.split_file}"
            )

        if class_mapping_file is None:
            class_mapping_file = (
                self.split_file.parent / "class_mapping.csv"
            )

        self.class_mapping_file = Path(class_mapping_file)

        if not self.class_mapping_file.is_file():
            raise FileNotFoundError(
                f"Class mapping not found: {self.class_mapping_file}"
            )

        # --------------------------------------------------
        # 1. Load the shared class mapping.
        # --------------------------------------------------
        self.class_to_idx = {}

        with self.class_mapping_file.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            reader = csv.DictReader(file)

            required_columns = {
                "ClassID", "ClassName", "OriginalID"
            }

            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    "class_mapping.csv must contain: "
                    "ClassID, ClassName, OriginalID"
                )

            for row in reader:
                class_name = row["ClassName"].strip()
                class_id = int(row["ClassID"])

                if class_name in self.class_to_idx:
                    raise ValueError(
                        f"Duplicate class in mapping: {class_name}"
                    )

                self.class_to_idx[class_name] = class_id

        if len(self.class_to_idx) != 100:
            raise ValueError(
                f"Expected 100 classes, found {len(self.class_to_idx)}"
            )

        if "Archery" in self.class_to_idx:
            raise ValueError(
                "Archery must not exist in the class mapping."
            )

        expected_ids = set(range(100))
        actual_ids = set(self.class_to_idx.values())

        if actual_ids != expected_ids:
            raise ValueError(
                "Class IDs must be continuous integers from 0 to 99."
            )

        self.classes = sorted(
            self.class_to_idx,
            key=self.class_to_idx.get
        )

        # Case-insensitive lookup for ZIP manifest class names.
        self.class_name_lookup = {
            name.casefold(): name
            for name in self.class_to_idx
        }

        # --------------------------------------------------
        # 2. Read the requested split file.
        # --------------------------------------------------
        self.split_samples = []
        seen_videos = set()

        with self.split_file.open(
            "r", encoding="utf-8-sig"
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip().replace("\\", "/")

                if not line:
                    continue

                parts = line.split("/")

                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid split entry at line {line_number}: "
                        f"{line}"
                    )

                class_name, video_name = parts

                if class_name not in self.class_to_idx:
                    raise ValueError(
                        f"Unknown or excluded class at line "
                        f"{line_number}: {class_name}"
                    )

                if not video_name.startswith("v_"):
                    raise ValueError(
                        f"Invalid video name at line {line_number}: "
                        f"{video_name}"
                    )

                video_key = f"{class_name}/{video_name}"

                if video_key in seen_videos:
                    raise ValueError(
                        f"Duplicate video in split file: {video_key}"
                    )

                seen_videos.add(video_key)
                self.split_samples.append({
                    "class_name": class_name,
                    "video_name": video_name,
                    "video_key": video_key,
                })

        if not self.split_samples:
            raise RuntimeError(
                f"No videos found in split file: {self.split_file}"
            )

        # --------------------------------------------------
        # 3. Index videos available inside ZIP archives.
        # --------------------------------------------------
        zip_files = sorted(self.root_dir.glob("*.zip"))

        if not zip_files:
            raise RuntimeError(
                f"No ZIP files found in: {self.root_dir}"
            )

        available_videos = {}

        for zip_path in zip_files:
            with zipfile.ZipFile(zip_path, "r") as archive:
                names = [
                    name.replace("\\", "/")
                    for name in archive.namelist()
                ]

                manifest_path = next(
                    (
                        name for name in names
                        if Path(name).name.lower() == "manifest.csv"
                    ),
                    None,
                )

                if manifest_path is None:
                    raise FileNotFoundError(
                        f"manifest.csv missing in {zip_path.name}"
                    )

                manifest_data = archive.read(
                    manifest_path
                ).decode("utf-8-sig")

                reader = csv.DictReader(
                    io.StringIO(manifest_data)
                )

                required_manifest_columns = {"cls", "video"}

                if not required_manifest_columns.issubset(
                    reader.fieldnames or []
                ):
                    raise ValueError(
                        f"Invalid manifest columns in {zip_path.name}"
                    )

                # Build a fast lookup of RGB first-frame paths.
                rgb_prefixes = {}

                for name in names:
                    if name.endswith("/rgb/000.jpg"):
                        prefix = name[:-len("rgb/000.jpg")]
                        video_name = prefix.rstrip("/").split("/")[-1]
                        rgb_prefixes[video_name] = prefix

                for row in reader:
                    manifest_class_name = row["cls"].strip()
                    video_name = row["video"].strip()

                    # Convert manifest name to canonical mapping name.
                    class_name = self.class_name_lookup.get(
                        manifest_class_name.casefold()
                    )

                    # Archery and any other unmapped class are excluded.
                    if class_name is None:
                        continue

                    if video_name not in rgb_prefixes:
                        continue

                    video_key = f"{class_name}/{video_name}"

                    if video_key in available_videos:
                        raise ValueError(
                            f"Video appears in multiple ZIPs: {video_key}"
                        )

                    available_videos[video_key] = {
                        "zip_path": zip_path,
                        "video_prefix": rgb_prefixes[video_name],
                    }

        # --------------------------------------------------
        # 4. Match split entries with actual ZIP videos.
        # --------------------------------------------------
        self.samples = []
        missing_videos = []

        for item in self.split_samples:
            video_key = item["video_key"]

            if video_key not in available_videos:
                missing_videos.append(video_key)
                continue

            sample = dict(item)
            sample.update(available_videos[video_key])
            self.samples.append(sample)

        if missing_videos:
            preview = "\n".join(missing_videos[:10])
            raise FileNotFoundError(
                f"{len(missing_videos)} split videos were not found "
                f"in the ZIP archives.\nFirst missing videos:\n{preview}"
            )

        if len(self.samples) != len(self.split_samples):
            raise RuntimeError(
                "Dataset indexing count does not match split file."
            )

        # --------------------------------------------------
        # 5. RGB normalization.
        # --------------------------------------------------
        self.rgb_mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(3, 1, 1)

        self.rgb_std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(3, 1, 1)

        print("\n===== UCF101 DATASET =====")
        print(f"Split file: {self.split_file.name}")
        print(f"ZIP files scanned: {len(zip_files)}")
        print(f"Videos in split: {len(self.split_samples)}")
        print(f"Videos indexed: {len(self.samples)}")
        print(f"Classes: {len(self.classes)}")
        print(f"Archery included: {'Archery' in self.class_to_idx}")
        print("Dataset indexing: PASSED")

    def __len__(self):
        return len(self.samples)

    def _read_image(self, archive, path):
        image_bytes = archive.read(path)

        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.copy()

    def _load_rgb(self, archive, video_prefix):
        frames = []

        for i in range(self.num_frames):
            image_path = f"{video_prefix}rgb/{i:03d}.jpg"

            image = self._read_image(archive, image_path)
            image = image.convert("RGB")
            image = image.resize(
                (self.img_size, self.img_size)
            )

            array = np.asarray(
                image, dtype=np.float32
            ).copy() / 255.0

            tensor = torch.from_numpy(
                array
            ).permute(2, 0, 1)

            tensor = (
                tensor - self.rgb_mean
            ) / self.rgb_std

            frames.append(tensor)

        return torch.stack(frames)

    def _load_flow(self, archive, video_prefix):
        frames = []

        for i in range(self.num_frames):
            flow_x_path = f"{video_prefix}flow_x/{i:03d}.jpg"
            flow_y_path = f"{video_prefix}flow_y/{i:03d}.jpg"

            image_x = self._read_image(archive, flow_x_path)
            image_y = self._read_image(archive, flow_y_path)

            flow_x = np.asarray(
                image_x.convert("L"),
                dtype=np.float32
            ).copy()

            flow_y = np.asarray(
                image_y.convert("L"),
                dtype=np.float32
            ).copy()

            if flow_x.shape != (self.img_size, self.img_size):
                raise ValueError(
                    f"Unexpected Flow X size: {flow_x.shape}"
                )

            if flow_y.shape != (self.img_size, self.img_size):
                raise ValueError(
                    f"Unexpected Flow Y size: {flow_y.shape}"
                )

            # Decode DIS flow using FLOW_BOUND = 20.
            flow_x = (flow_x / 255.0) * 40.0 - 20.0
            flow_y = (flow_y / 255.0) * 40.0 - 20.0

            flow = np.stack(
                [flow_x, flow_y], axis=0
            )

            # Normalize optical flow approximately to [-1, 1].
            flow = flow / 20.0

            frames.append(torch.from_numpy(flow))

        return torch.stack(frames)

    def __getitem__(self, index):
        sample_info = self.samples[index]

        with zipfile.ZipFile(
            sample_info["zip_path"], "r"
        ) as archive:
            rgb = self._load_rgb(
                archive, sample_info["video_prefix"]
            )

            flow = self._load_flow(
                archive, sample_info["video_prefix"]
            )

        class_name = sample_info["class_name"]

        return {
            "rgb": rgb,
            "flow": flow,
            "label": torch.tensor(
                self.class_to_idx[class_name],
                dtype=torch.long
            ),
            "video_name": sample_info["video_name"],
            "class_name": class_name,
        }
