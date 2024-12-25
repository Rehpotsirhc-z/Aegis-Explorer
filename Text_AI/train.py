from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.onnx


class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        data_path = Path(data_path) if isinstance(data_path, str) else data_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data = []
        self.labels = []

        categories = [
            "good",
            "drugs",
            "explicit",
            "gambling",
            "games",
            # "monetary",
            "profanity",
            # "social",
        ]
        category_to_id = {className: id for id, className in enumerate(categories)}

        for category in categories:
            category_dir = data_path / category
            for file_path in category_dir.iterdir():
                if file_path.is_file():
                    text = file_path.read_text(encoding="utf-8").strip().lower()
                    self.data.append(text)
                    self.labels.append(category_to_id[category])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        return (
            inputs["input_ids"].squeeze(0),
            inputs["attention_mask"].squeeze(0),
            torch.tensor(label),
        )


def save_onnx(model, tokenizer, output_path, device):
    """
    Exports the model to ONNX format.
    
    Args:
        model (torch.nn.Module): The PyTorch model to export.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for generating dummy inputs.
        output_path (str or Path): The file path to save the exported ONNX model.
        device (torch.device): The device (CPU or GPU) on which the model is running.
    """
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Ensure the model is on the correct device

    # Create dummy input for export
    dummy_text = "This is a sample input."
    dummy_inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True)
    dummy_input_ids = dummy_inputs["input_ids"].to(device)
    dummy_attention_mask = dummy_inputs["attention_mask"].to(device)

    try:
        # Export the model
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),  # Model inputs
            output_path,                              # File path to save ONNX model
            input_names=["input_ids", "attention_mask"],  # Input names
            output_names=["logits"],                    # Output names
            dynamic_axes={
                "input_ids": {0: "batch_size"},  # Variable batch size
                "attention_mask": {0: "batch_size"},
            },
            opset_version=11,  # Choose an appropriate ONNX opset version
        )
        print(f"Model successfully exported to ONNX at {output_path}")
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")


def combine_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)

    return input_ids, attention_mask, labels


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            # Move inputs and labels to the specified device (GPU or CPU)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Ensure model is on the same device
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Move predictions back to CPU for metric calculation
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics on CPU
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")

    return accuracy, precision, recall, f1



def train(train_dir, val_dir, model_dir, batch_size=16, epochs=60, learning_rate=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=6
    ).to(device)

    train_dataset = TextDataset(train_dir, tokenizer)
    val_dataset = TextDataset(val_dir, tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=combine_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=combine_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model_dir = Path(model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            # Move inputs and labels to the GPU
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss}")

    torch.save(model.state_dict(), model_dir)
    print(f"Model saved to {model_dir}")

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate(model, val_loader, device)
    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation Precision: {precision}")
    print(f"Validation Recall: {recall}")
    print(f"Validation F1: {f1}")

    # Save the model to ONNX format
    save_onnx(model, tokenizer, Path("model/model.onnx"), device)


if __name__ == "__main__":
    train("dataset/train", "dataset/validation", "model/model.pth")
