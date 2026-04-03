"""Create fixed train/val/test split files shared by all baseline scripts."""

from pathlib import Path

from dataset_utils import create_or_load_fixed_splits, resolve_default_data_dir


def main():
    data_dir = resolve_default_data_dir()
    split_dir = Path(__file__).resolve().parent / "splits"

    train_records, val_records, test_records = create_or_load_fixed_splits(
        data_dir=str(data_dir),
        split_dir=str(split_dir),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print(f"Data dir: {data_dir}")
    print(f"Split dir: {split_dir}")
    print(f"Train: {len(train_records)}")
    print(f"Val: {len(val_records)}")
    print(f"Test: {len(test_records)}")


if __name__ == "__main__":
    main()
