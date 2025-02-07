import os
import shutil
import argparse
import math
import random
from pathlib import Path
import logging
import colorlog

formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)


logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Ratio:
    train: float
    validation: float
    test: float

    def __init__(self, train_ratio=None, validation_ratio=None, test_ratio=None):
        if train_ratio is None and validation_ratio is None and test_ratio is None:
            self.train = 0.7
            self.validation = 0.2
            self.test = 0.1
        else:
            ratios = {
                "train": train_ratio,
                "validation": validation_ratio,
                "test": test_ratio,
            }
            defined_ratios = {k: v for k, v in ratios.items() if v is not None}

            if len(defined_ratios) == 3:
                self.train = defined_ratios.get("train")
                self.validation = defined_ratios.get("validation")
                self.test = defined_ratios.get("test")
            elif len(defined_ratios) == 2:
                ratiosum = sum(defined_ratios.values())
                missing_ratio = 1 - ratiosum

                for key in ["train", "validation", "test"]:
                    if key not in defined_ratios:
                        setattr(self, key, missing_ratio)
                    else:
                        setattr(self, key, defined_ratios[key])
            else:
                raise ValueError("At least two ratios are needed")

        self._validate_ratios()

    def _validate_ratios(self):
        total = self.train + self.validation + self.test
        tolerance = 1e-9
        if abs(total - 1) > tolerance:
            raise ValueError(f"Ratios must sum to 1, not {total}")
        if any(r < 0 for r in [self.train, self.validation, self.test]):
            raise ValueError("Ratios must be non-negative")


def generate_label_file(img_dir, dest_dir, class_id):
    img_dir = Path(img_dir).absolute() if isinstance(img_dir, str) else img_dir
    dest_dir = Path(dest_dir).absolute() if isinstance(dest_dir, str) else dest_dir
    label_file_path = dest_dir / img_dir.with_suffix(".txt").name

    label_file_path.write_text(f"{class_id} 0.5 0.5 1 1\n")


def split_dataset(src_dir, dest_dir, ratios=Ratio()):
    src_dir = Path(src_dir) if isinstance(src_dir, str) else src_dir
    dest_dir = Path(dest_dir) if isinstance(dest_dir, str) else dest_dir

    train_ratio = ratios.train
    validation_ratio = ratios.validation
    test_ratio = ratios.test

    # test_ratio = 1 - (train_ratio + validation_ratio)
    # if not (0 <= train_ratio <= 1 and 0 <= validation_ratio <= 1 and test_ratio >= 0):
    #     print("Invalid ratio")
    #     return

    # Create destination directories
    for segment in ["train", "validation", "test"]:
        segment_dir = dest_dir / segment
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "background",
        "drugs",
        "explicit",
        "gambling",
        "games",
        # "monetary",
        "profanity",
        "good",
        # "social",
    ]
    class_to_id = {class_name: idx for idx, class_name in enumerate(class_names)}

    img_extensions = [".jpg", ".jpeg", ".png", ".bmp", "webp", ".tiff", ".avif"]

    # Iterate over each class folder
    for class_folder in src_dir.iterdir():
        if class_folder.is_dir():
            all_files = [
                f for f in class_folder.iterdir() if f.suffix.lower() in img_extensions
            ]
            # all_files = [f for f in class_folder.iterdir() if f.suffix == ".jpg"]
            random.shuffle(all_files)

            train_num = int(len(all_files) * train_ratio)
            validation_num = int(len(all_files) * validation_ratio)
            test_num = len(all_files) - train_num - validation_num
            logger.debug(f"{class_folder}: ({train_num}, {validation_num}, {test_num})")

            train_files = all_files[:train_num]
            validation_files = all_files[train_num : train_num + validation_num]
            test_files = all_files[train_num + validation_num :]

            for files, split in [
                (train_files, "train"),
                (validation_files, "validation"),
                (test_files, "test"),
            ]:
                output_dir = dest_dir / split / class_folder.name
                output_dir.mkdir(parents=True, exist_ok=True)

                for file in files:
                    source_path = file
                    destination_path = output_dir / file.name
                    logger.debug(f"{source_path} -> {destination_path}")
                    shutil.copy(source_path, destination_path)
                    generate_label_file(
                        source_path, output_dir, class_to_id[class_folder.name]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # prog="",
        # description="",
        # formatter_class=ColorHelpFormatter,
        add_help=True,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="be verbose")
    parser.add_argument(
        "source_dir",
        nargs="?",
        default="images",
        help="source directory (default: `images`)",
    )
    parser.add_argument(
        "destination_dir",
        nargs="?",
        default="training_data",
        help="destination directory (default: `training_data`)",
    )

    args = parser.parse_args()

    source_dir = Path(args.source_dir).absolute()
    destination_dir = Path(args.destination_dir).absolute()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    split_dataset(source_dir, destination_dir)
