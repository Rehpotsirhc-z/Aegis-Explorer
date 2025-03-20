import torch
from torch.utils.data import Dataset, random_split
import re
import os

categories = ["background", "drugs", "explicit", "gambling", "games", "profanity"] # TODO Maybe add background

class BannedWordDataset(Dataset):
    banned_words: dict

    def __init__(self, banned_dir, corpus_file, vocab=None, max_len=20):
        """
        Reads banned words from txt files in banned_dir and extracts words from corpus_file.
        Each word is labeled by the index of the category if it is banned and 0 (background) otherwise.
        A simple character-level representation is used.

        Multi-word phrases in the banned files are preserved.
        """
        self.banned_words = dict()

        for text_file in os.listdir(banned_dir):
            # Load banned words (assumed to be one per line, already lowercased)
            with open(os.path.join(banned_dir, text_file), 'r', encoding='utf-8') as f:
                category = os.path.splitext(os.path.basename(text_file))[0]
                # print(category)
                self.banned_words[category] = set(line.strip().lower() for line in f if line.strip().lower())

        # Read the corpus and extract words
        corpus_words = []
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Use regex to extract words and convert to lowercase
                words = re.findall(r'\w+', line.lower())
                corpus_words.extend(words)
                
        # Create examples: label = category index for banned, 0 otherwise
        self.examples = []
        for word in corpus_words:
            label = 0  # Default to background
            for idx, category in enumerate(categories):
                if word in self.banned_words.get(category, set()):
                    label = idx
                    # print(label)
                    break

            self.examples.append((word, label))
            
        # Ensure that all banned words are included (in case they never appear in the corpus)
        # for word in self.banned_words:
        #     word_label = 0
        #     for idx, category in enumerate(categories):
        #         if word in self.banned_words.get(category, set()):
        #             word_label = idx
        #             break
        #     self.examples.append((word, word_label))
        for cat, banned_set in self.banned_words.items():
            for term in banned_set:
                label = categories.index(cat) if cat in categories else 0
                self.examples.append((term, label))
            
        self.max_len = max_len

        # Build or load vocabulary
        if vocab is None:
            self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self):
        # Create a character-level vocabulary from the dataset words
        chars = set()
        for word, _ in self.examples:
            chars.update(list(word))
        # Reserve index 0 for padding and 1 for unknown characters
        self.vocab = {'<pad>': 0, '<unk>': 1}
        for char in sorted(chars):
            self.vocab[char] = len(self.vocab)

    def encode_word(self, word):
        # Convert a word into a list of character indices
        seq = [self.vocab.get(c, self.vocab['<unk>']) for c in word]
        # Pad with the <pad> token or truncate to max_len
        if len(seq) < self.max_len:
            seq += [self.vocab['<pad>']] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        word, label = self.examples[idx]
        word_encoded = self.encode_word(word)
        return word_encoded, torch.tensor(label, dtype=torch.long)


def get_datasets(banned_dir, corpus_file, train_ratio=0.8, max_len=20):
    """
    Returns training and validation datasets.
    """
    dataset = BannedWordDataset(banned_dir, corpus_file, max_len=max_len)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
