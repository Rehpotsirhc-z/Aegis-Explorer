import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_datasets, CATEGORIES
import time

# Hyperparameters
MAX_LEN = 256
EMBEDDING_DIM = 100
NUM_CLASSES = 1 + len(CATEGORIES)  # background (0) + others

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, max_len):
        """
        A 1D CNN for character-level sequence labeling.
        Input: sequence of character indices of length max_len.
        Output: per-character logits of shape [max_len, num_classes].
        """
        super(CNNClassifier, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_out = nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        embedded = self.embedding(x)             # [batch, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)       # [batch, embedding_dim, seq_len]
        out = self.conv1(embedded)               # [batch, 128, seq_len]
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)                    # [batch, 128, seq_len]
        out = torch.relu(out)
        out = self.dropout(out)
        logits = self.conv_out(out)              # [batch, num_classes, seq_len]
        logits = logits.permute(0, 2, 1)           # [batch, seq_len, num_classes]
        return logits