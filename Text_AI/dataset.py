import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize

categories = ["drugs", "explicit", "gambling", "games", "profanity", "good"]
# categories = ["drugs", "explicit", "gambling", "games", "good", "monetary", "profanity", "social"]


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


def split_dataset(src_dir, dest_dir, ratio=Ratio()):
    src_dir = Path(src_dir) if isinstance(src_dir, str) else src_dir
    dest_dir = Path(dest_dir) if isinstance(dest_dir, str) else dest_dir

    train_ratio = ratio.train
    validation_ratio = ratio.validation
    test_ratio = ratio.test

    for category in categories:
        file_path = src_dir / f"{category}.txt"

        # Read and tokenize the content of the file into sentences
        content = file_path.read_text(encoding="utf-8").strip().lower()
        sentences = sent_tokenize(content)

        print(f"Category: {category}")

        train, temp = train_test_split(sentences, train_size=train_ratio)
        validation, test = train_test_split(
            temp, train_size=validation_ratio / (validation_ratio + test_ratio)
        )

        def save_sentences(sentences, file_path):
            for i, sentence in enumerate(sentences):
                filename = file_path / f"{category}_{i}.txt"
                filename.write_text(sentence, encoding="utf-8")

        save_sentences(train, dest_dir / "train" / category)
        save_sentences(validation, dest_dir / "validation" / category)
        save_sentences(test, dest_dir / "test" / category)


destination = Path("dataset")
train_dir = destination / "train"
validation_dir = destination / "validation"
test_dir = destination / "test"

for dir in [train_dir, validation_dir, test_dir]:
    for category in categories:
        path = dir / category
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(exist_ok=True, parents=True)


split_dataset("alt_source", destination)
