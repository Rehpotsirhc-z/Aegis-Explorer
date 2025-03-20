import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import get_datasets

categories = ["background", "drugs", "explicit", "gambling", "games", "profanity"] # TODO Maybe add background

class BannedWordClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64, output_dim=6):
        """
        A simple character-level classifier that uses an embedding layer, a GRU,
        and a fully connected layer for banned word classification.
        Output: 0=background, 1=drugs, 2=explicit, 3=gambling, 4=games, 5=profanity
        """
        super(BannedWordClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)         # (batch_size, seq_len, embedding_dim)
        _, h = self.gru(embedded)              # h shape: (1, batch_size, hidden_dim)
        h = h.squeeze(0)                       # (batch_size, hidden_dim)
        out = self.fc(h)                       # (batch_size, output_dim)
        return out

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    """
    Trains the model and prints training loss and validation accuracy per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Evaluate on the validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
    
    return model

if __name__ == "__main__":
    # File paths to your banned words list and corpus
    banned_dir = "source"
    corpus_file = "corpus.txt"
    
    # Create datasets and corresponding DataLoaders
    train_dataset, val_dataset = get_datasets(banned_dir, corpus_file, train_ratio=0.8, max_len=20)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Retrieve vocabulary size from the underlying dataset object
    vocab_size = len(train_dataset.dataset.vocab)
    
    # Initialize the model
    model = BannedWordClassifier(vocab_size, output_dim=len(categories))
    
    # Train the model
    trained_model = train_model(model, train_loader, val_loader, epochs=30, lr=1e-3)
    
    # Save the standard trained model weights (if needed)
    torch.save(trained_model.state_dict(), "banned_classifier.pt")
    
    # Optimize the model using TorchScript for faster inference
    scripted_model = torch.jit.script(trained_model)
    scripted_model.save("banned_classifier_torchscript.pt")
    print("Saved TorchScript model.")

    # Optionally, perform dynamic quantization if running on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quantized_model = torch.quantization.quantize_dynamic(
        trained_model, {nn.Linear, nn.GRU}, dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), "banned_classifier_quantized.pt")
    print("Saved dynamically quantized model.")
