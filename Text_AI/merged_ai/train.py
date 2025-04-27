import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_datasets, CATEGORIES
import time

# Hyperparameters
MAX_LEN = 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100
NUM_CLASSES = 1 + len(CATEGORIES)  # background (0) + others
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

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

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    class_weights = torch.tensor([0.05] + [1.0]*len(CATEGORIES), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Training started at {current_time} on {device}")

    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), y_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), y_batch.reshape(-1))
                val_loss += loss.item() * x_batch.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"{current_time}: Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "cnn_model.pt")
    return model

if __name__ == "__main__":
    banned_dir = "banned"
    corpus_file = "corpus.txt"
    train_dataset, val_dataset = get_datasets(banned_dir, corpus_file, train_ratio=0.8, max_len=MAX_LEN)
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    vocab_size = len(train_dataset.dataset.vocab)
    model = CNNClassifier(vocab_size, EMBEDDING_DIM, NUM_CLASSES, MAX_LEN)
    print(f"Model architecture: {model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, device)

    # Save a TorchScript version too.
    torchscript_model = torch.jit.script(trained_model)
    torchscript_model.save("cnn_model_scripted.pt")
