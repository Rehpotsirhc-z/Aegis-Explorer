from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from dataset import BannedWordDataset, CATEGORIES  # our dataset file
from train import CNNClassifier, MAX_LEN, EMBEDDING_DIM, NUM_CLASSES
import json

app = Flask(__name__)

# Configuration
BANNED_DIR = "banned"
ALLOWED_WORDS = "whitelist.txt"
ALLOWED_WORDS_SET = set()

# Load allowed words from the whitelist file
try:
    with open(ALLOWED_WORDS, 'r') as f:
        for line in f:
            word = line.strip().lower()
            if word:  # Ensure it's not empty
                ALLOWED_WORDS_SET.add(word)
except FileNotFoundError:
    print(f"Whitelist file missing")

CORPUS_FILE = "corpus.txt"
MAX_LEN = 256
CONFIDENCE_THRESHOLD = 0.99  # if max probability is below this, mark as background

# Load the dataset (to get vocabulary and label mapping)
dataset_obj = BannedWordDataset(BANNED_DIR, CORPUS_FILE, max_len=MAX_LEN)
vocab = dataset_obj.vocab
label_to_category = dataset_obj.label_to_category

# Build model.
vocab_size = len(vocab)
model = CNNClassifier(vocab_size, EMBEDDING_DIM, NUM_CLASSES, MAX_LEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("cnn_model.pt", map_location=device))
model.to(device)
model.eval()

def encode_text(text, vocab, max_len):
    """Encode input text to a list of indices, pad/truncate to max_len."""
    text = text.lower()
    indices = [vocab.get(ch, vocab["<UNK>"]) for ch in text]
    if len(indices) < max_len:
        pad_len = max_len - len(indices)
        indices.extend([vocab["<PAD>"]] * pad_len)
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)

def decode_labels(label_seq, label_to_category, min_span_length=2):
    """
    Given a list of predicted labels (length = max_len),
    post-process to extract contiguous spans where label != 0.
    Returns a list of dicts: {start: int, end: int, category: str}
    """
    spans = []
    current_span = None
    for i, label in enumerate(label_seq):
        if label != 0:
            cat = label_to_category.get(label, "unknown")
            if current_span is None:
                current_span = {"start": i, "end": i, "category": cat}
            else:
                if current_span["category"] == cat:
                    current_span["end"] = i
                else:
                    spans.append(current_span)
                    current_span = {"start": i, "end": i, "category": cat}
        else:
            if current_span is not None:
                spans.append(current_span)
                current_span = None
    if current_span is not None:
        spans.append(current_span)
    # Filter out spans shorter than min_span_length.
    spans = [span for span in spans if (span["end"] - span["start"] + 1) >= min_span_length]
    return spans

def expand_to_word_boundaries(text, start, end):
    """
    Expands a given span [start, end] to cover the entire word in text.
    A word is defined as a contiguous sequence of non-whitespace characters.
    Returns (new_start, new_end) indices.
    """
    # Expand to the left.
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    # Expand to the right.
    while end < len(text) - 1 and not text[end + 1].isspace():
        end += 1
    return start, end

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Provide a JSON payload with a 'texts' key containing a list of texts."}), 400
    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list."}), 400

    # Encode texts into tensor batch.
    batch = []
    for text in texts:
        encoded = encode_text(text, vocab, MAX_LEN)
        batch.append(encoded)
    batch = torch.stack(batch)  # [batch_size, MAX_LEN]
    batch = batch.to(device)

    with torch.no_grad():
        logits = model(batch)  # [batch_size, MAX_LEN, NUM_CLASSES]
        probs = torch.softmax(logits, dim=-1)  # [batch_size, MAX_LEN, NUM_CLASSES]
        max_probs, preds = torch.max(probs, dim=-1)  # both: [batch_size, MAX_LEN]
        # If confidence is low, set the prediction to background (0).
        # print(max_probs)
        preds[max_probs < CONFIDENCE_THRESHOLD] = 0

    preds = preds.cpu().tolist()
    results = []
    for text, pred_seq in zip(texts, preds):
        effective_length = min(len(text), MAX_LEN)
        pred_seq = pred_seq[:effective_length]
        spans = decode_labels(pred_seq, label_to_category, min_span_length=3)

        for span in spans:
            original_phrase = text[span["start"]:span["end"] + 1]
            if original_phrase.strip().lower() in ALLOWED_WORDS_SET:
                spans.remove(span)  # Remove this span if it's in the whitelist

        # Expand spans to word boundaries
        for span in spans:
            start, end = span["start"], span["end"]
            new_start, new_end = expand_to_word_boundaries(text, start, end)
            span["start"], span["end"] = new_start, new_end
            
            new_phrase = text[new_start:new_end + 1]
            if new_phrase.strip().lower() in ALLOWED_WORDS_SET:
                spans.remove(span)

        results.append({"text": text, "flags": spans})
    return jsonify(results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
