from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.onnx
from train import TextDataset, combine_fn, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load the text ai
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.load_state_dict(torch.load('model/model_v7.pth', map_location=device), strict=False)

model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

test_dataset = TextDataset("dataset/test", tokenizer)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=combine_fn)

accuracy, precision, recall, f1 = evaluate(model, test_loader, device)

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1: {f1}")