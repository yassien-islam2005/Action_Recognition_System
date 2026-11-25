import csv
import json
import re
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT / "data/splits/ucf101/split_1"

def read_list(path):
    return [
        line.strip().replace("\\", "/").removesuffix(".avi")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

def parse_video(path):
    parts = path.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid path: {path}")

    cls, video = parts
    match = re.search(r"_g(\d+)_c\d+$", video)

    if not match:
        raise ValueError(f"Cannot extract group: {video}")

    return cls, f"{cls}_g{match.group(1)}"

def check_unique(items, name):
    duplicates = [x for x, n in Counter(items).items() if n > 1]
    if duplicates:
        raise ValueError(f"{name}: {len(duplicates)} duplicates")

def main():
    train_all = read_list(SPLIT_DIR / "train_official.txt")
    test = read_list(SPLIT_DIR / "test_official.txt")

    with open(SPLIT_DIR / "class_mapping.csv", encoding="utf-8-sig") as f:
        mapping = list(csv.DictReader(f))

    classes = {row["ClassName"] for row in mapping}

    print("===== INPUT AUDIT =====")
    print("Classes:", len(classes))
    print("Official train:", len(train_all))
    print("Official test:", len(test))

    if len(classes) != 100 or "Archery" in classes:
        raise ValueError("Invalid class mapping")

    if any(x.startswith("Archery/") for x in train_all + test):
        raise ValueError("Archery found in input lists")

    check_unique(train_all, "Official train")
    check_unique(test, "Official test")

    if set(train_all) & set(test):
        raise ValueError("Official train/test video overlap")

    train_meta = [parse_video(x) for x in train_all]
    test_meta = [parse_video(x) for x in test]

    y = [cls for cls, group in train_meta]
    groups = [group for cls, group in train_meta]

    test_groups = {group for cls, group in test_meta}
    train_official_groups = set(groups)

    if train_official_groups & test_groups:
        raise ValueError("Official train/test group overlap")

    if set(y) != classes:
        raise ValueError("Train classes do not match mapping")

    if {cls for cls, group in test_meta} != classes:
        raise ValueError("Test classes do not match mapping")

    print("\n===== GROUP-AWARE SPLIT =====")

    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    best = None
    X = [[0] for _ in train_all]

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups), 1
    ):
        candidate_train = [train_all[i] for i in train_idx]
        candidate_val = [train_all[i] for i in val_idx]

        train_counts = Counter(parse_video(x)[0] for x in candidate_train)
        val_counts = Counter(parse_video(x)[0] for x in candidate_val)

        if set(train_counts) != classes or set(val_counts) != classes:
            continue

        train_groups = {parse_video(x)[1] for x in candidate_train}
        val_groups = {parse_video(x)[1] for x in candidate_val}

        if train_groups & val_groups:
            raise ValueError("Train/validation group leakage")

        ratio = len(candidate_val) / len(train_all)

        class_errors = [
            abs(val_counts[c] / (val_counts[c] + train_counts[c]) - 0.10)
            for c in classes
        ]

        score = abs(ratio - 0.10) + sum(class_errors) / len(class_errors)

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "fold": fold,
                "train": candidate_train,
                "val": candidate_val,
                "train_groups": train_groups,
                "val_groups": val_groups,
                "ratio": ratio
            }

    if best is None:
        raise ValueError("No valid fold contains all 100 classes")

    final_train = best["train"]
    val = best["val"]

    print("Selected fold:", best["fold"])
    print("Validation ratio:", round(best["ratio"] * 100, 2), "%")

    print("\n===== FINAL AUDIT =====")

    check_unique(final_train, "Train")
    check_unique(val, "Validation")
    check_unique(test, "Test")

    train_set, val_set, test_set = map(set, [final_train, val, test])

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Video overlap detected")

    if best["train_groups"] & best["val_groups"]:
        raise ValueError("Train/validation group overlap")

    if best["train_groups"] & test_groups:
        raise ValueError("Train/test group overlap")

    if best["val_groups"] & test_groups:
        raise ValueError("Validation/test group overlap")

    for name, items in [
        ("Train", final_train),
        ("Validation", val),
        ("Test", test)
    ]:
        found = {parse_video(x)[0] for x in items}
        if found != classes:
            raise ValueError(f"{name} missing classes")

    print("Train:", len(final_train))
    print("Validation:", len(val))
    print("Test:", len(test))
    print("Total:", len(final_train) + len(val) + len(test))
    print("Train classes:", len({parse_video(x)[0] for x in final_train}))
    print("Validation classes:", len({parse_video(x)[0] for x in val}))
    print("Test classes:", len({parse_video(x)[0] for x in test}))
    print("Video overlap: 0")
    print("Group overlap: 0")
    print("Archery: excluded")

    outputs = {
        "train.txt": final_train,
        "val.txt": val,
        "test.txt": test
    }

    for filename, items in outputs.items():
        (SPLIT_DIR / filename).write_text(
            "\n".join(items) + "\n",
            encoding="utf-8"
        )

    info = {
        "dataset": "UCF101",
        "classes": 100,
        "excluded_classes": ["Archery"],
        "official_split": 1,
        "method": "StratifiedGroupKFold",
        "n_splits": 10,
        "selected_fold": best["fold"],
        "random_state": 42,
        "target_validation_ratio": 0.10,
        "actual_validation_ratio": best["ratio"],
        "train_videos": len(final_train),
        "validation_videos": len(val),
        "test_videos": len(test),
        "total_videos": len(final_train) + len(val) + len(test),
        "grouping": "class_name + gXX"
    }

    (SPLIT_DIR / "split_info.json").write_text(
        json.dumps(info, indent=4),
        encoding="utf-8"
    )

    report = [
        "UCF101 100-CLASS SPLIT AUDIT",
        "============================",
        f"Train videos: {len(final_train)}",
        f"Validation videos: {len(val)}",
        f"Test videos: {len(test)}",
        "Classes in each split: 100",
        "Video overlap: 0",
        "Group overlap: 0",
        "Archery excluded: YES",
        "Official test preserved: YES",
        "Method: StratifiedGroupKFold",
        f"Selected fold: {best['fold']}",
        "Final status: PASSED"
    ]

    (SPLIT_DIR / "audit_report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\n===== SAVED FILES =====")
    for name in [
        "train.txt", "val.txt", "test.txt",
        "split_info.json", "audit_report.txt"
    ]:
        print(SPLIT_DIR / name)

    print("\nSPLIT GENERATION: PASSED")

if __name__ == "__main__":
    main()
