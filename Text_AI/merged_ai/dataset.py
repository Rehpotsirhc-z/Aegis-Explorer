import os
import re
import time
import torch
import pickle
from torch.utils.data import Dataset, random_split

# Define categories.
# Label 0 is reserved for background.
CATEGORIES = ["drugs", "explicit", "gambling", "games", "profanity"]
# categories = ["drugs", "explicit", "gambling", "games", "background", "monetary", "profanity", "social"]

CACHE_FILENAME = "preprocessed_dataset.pkl"

class BannedWordDataset(Dataset):
    def __init__(self, banned_dir, corpus_file, max_len=256, vocab=None):
        """
        Reads banned phrases from text files in banned_dir and reads sentences from corpus_file.
        Each sentence is lowercased and then scanned for banned phrases using combined regex patterns
        that enforce word boundaries.
        Each character in a banned phrase is labeled with the corresponding category ID;
        all other characters are labeled 0 (background).
        Sentences are represented as a list of characters and padded/truncated to max_len.
        If vocab is not provided, builds a character vocabulary from the corpus and banned phrases.
        Also caches preprocessed samples to speed up subsequent runs.
        """
        self.max_len = max_len

        # Build banned phrases mapping: category -> list of phrases.
        self.banned = {}
        for filename in os.listdir(banned_dir):
            if not filename.endswith(".txt"):
                continue
            category = os.path.splitext(filename)[0].lower()
            if category not in CATEGORIES:
                continue
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"{current_time}: Loading banned phrases for category: {category}")
            filepath = os.path.join(banned_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                phrases = [line.strip().lower() for line in f if line.strip()]
                self.banned[category] = phrases

        # Create category label mapping.
        self.category_to_label = {cat: i+1 for i, cat in enumerate(CATEGORIES)}
        self.label_to_category = {0: "background"}
        for cat, label in self.category_to_label.items():
            self.label_to_category[label] = cat

        # Combine banned phrases into one regex per category using word boundaries.
        self.banned_patterns = {}  # category -> compiled regex pattern.
        for cat, phrases in self.banned.items():
            # Using non-capturing group and word boundaries.
            combined_pattern = r'\b(?:' + "|".join(re.escape(phrase) for phrase in phrases) + r')\b'
            self.banned_patterns[cat] = re.compile(combined_pattern)

        # Try to load preprocessed samples from cache.
        if os.path.exists(CACHE_FILENAME):
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"{current_time}: Loading cached dataset from {CACHE_FILENAME}...")
            with open(CACHE_FILENAME, 'rb') as cache_file:
                self.samples = pickle.load(cache_file)
            print(f"Loaded cached dataset with {len(self.samples)} samples.")
        else:
            self.samples = []  # list of (list_of_chars, label_seq)
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"{current_time}: Reading corpus file...")
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    sentence = line.strip().lower()
                    if not sentence:
                        continue
                    # Convert sentence to a list of characters.
                    char_list = list(sentence)
                    labels = [0] * len(char_list)
                    # Mark banned phrase spans using combined regex.
                    for cat, pattern in self.banned_patterns.items():
                        label_val = self.category_to_label.get(cat, 0)
                        for match in pattern.finditer(sentence):
                            start, end = match.start(), match.end()
                            for i in range(start, end):
                                if labels[i] == 0 or label_val < labels[i]:
                                    labels[i] = label_val
                    # Pad or truncate.
                    if len(char_list) < max_len:
                        pad_length = max_len - len(char_list)
                        char_list.extend(["<PAD>"] * pad_length)
                        labels.extend([0] * pad_length)
                    else:
                        char_list = char_list[:max_len]
                        labels = labels[:max_len]
                    self.samples.append((char_list, labels))
            print("Dataset loaded. Number of samples:", len(self.samples))
            # Cache the preprocessed dataset.
            with open(CACHE_FILENAME, 'wb') as cache_file:
                pickle.dump(self.samples, cache_file)
            print(f"Cached dataset to {CACHE_FILENAME}.")

        # Build or load vocabulary.
        if vocab is None:
            self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self):
        print("Building vocabulary...")
        chars = set()
        for char_list, _ in self.samples:
            chars.update(char_list)
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for ch in sorted(chars):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

    def encode_text(self, char_list):
        return [self.vocab.get(ch, self.vocab["<UNK>"]) for ch in char_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        char_list, labels = self.samples[idx]
        text_indices = self.encode_text(char_list)
        x = torch.tensor(text_indices, dtype=torch.long)  # [max_len]
        y = torch.tensor(labels, dtype=torch.long)        # [max_len]
        return x, y

def get_datasets(banned_dir, corpus_file, train_ratio=0.8, max_len=256):
    dataset = BannedWordDataset(banned_dir, corpus_file, max_len=max_len)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
