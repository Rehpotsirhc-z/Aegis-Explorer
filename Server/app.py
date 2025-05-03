import os
import time
from pathlib import Path
from flask import Flask, request, jsonify
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BertTokenizerFast,
    BertForTokenClassification,
)
from flask_cors import CORS
from ultralytics import YOLO
import torch
from PIL import Image
from io import BytesIO
from torch.quantization import quantize_dynamic
from dataset import BannedWordDataset
from text_model import CNNClassifier, MAX_LEN, EMBEDDING_DIM, NUM_CLASSES
import difflib
import json

# set number of threads to 8
# torch.set_num_threads(8)

WHITELIST_FILE = "whitelist.txt"
BANNED_DIR = "banned"
ALLOWED_WORDS_SET = set()
# Load allowed words from the whitelist file
try:
    with open(WHITELIST_FILE, 'r') as f:
        for line in f:
            word = line.strip().lower()
            if word:  # Ensure it's not empty
                ALLOWED_WORDS_SET.add(word)
except FileNotFoundError:
    print(f"Whitelist file missing")
BLACKLIST_FILE = "blacklist.txt"
BLACKLIST_JSON_FILE = "blacklist.json"

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
image_model_path = Path("models/image/model_v9.pt")
img_model = YOLO(image_model_path)
img_model.to(device)

# Load supplementary model
# SUPPLEMENTARY_MODEL_PATH = "models/results"
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# text_supp_model = BertForTokenClassification.from_pretrained(SUPPLEMENTARY_MODEL_PATH)
# text_supp_model.eval()

CORPUS_FILE = "corpus.txt"
MAX_LEN = 256
CONFIDENCE_THRESHOLD = 0.99  # if max probability is below this, mark as background

# Load the dataset (to get vocabulary and label mapping)
dataset_obj = BannedWordDataset(BANNED_DIR, CORPUS_FILE, max_len=MAX_LEN)
vocab = dataset_obj.vocab
label_to_category = dataset_obj.label_to_category

# Build model.
vocab_size = len(vocab)
supp_text_model = CNNClassifier(vocab_size, EMBEDDING_DIM, NUM_CLASSES, MAX_LEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
supp_text_model.load_state_dict(torch.load("cnn_model.pt", map_location=device))
supp_text_model.to(device)
supp_text_model.eval()

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=6
)

text_model.load_state_dict(
    torch.load("models/text/model_v7.pth", map_location=device),
    strict=False,
)

# text_model = quantize_dynamic(text_model, {torch.nn.Linear}, dtype=torch.qint8)
text_model.to(device)


text_model.eval()


def load_word_set(file_path):
    word_set = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            word_set = {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Warning: {file_path} not found; ignoring.")
    return word_set


banned_words = load_word_set(BLACKLIST_FILE)
banned_dict = json.load(open(BLACKLIST_JSON_FILE, "r", encoding="utf-8")) if Path(BLACKLIST_JSON_FILE).exists() else {}


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

def decode_labels(label_seq, token_confidences, label_to_category, min_span_length=3):
    """
    Given a list of predicted labels (length = max_len),
    post-process to extract contiguous spans where label != 0.
    Returns a list of dicts: {start: int, end: int, category: str, confidence: float}.
    """
    spans = []
    current_span = None
    for i, label in enumerate(label_seq):
        if label != 0:
            cat = label_to_category.get(label, "unknown")
            # Get confidence for the current token:
            confidence = token_confidences[i]

            print(f"Token: {label}, Confidence: {confidence}")
            if current_span is None:
                current_span = {"start": i, "end": i, "category": cat}
                current_span["confidence"] = confidence
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

    bounded_start = min(start, len(text)-1)
    bounded_end = max(end, 0)

    # Expand to the left.
    while bounded_start > 0 and not text[bounded_start - 1].isspace():
        bounded_start -= 1
    # Expand to the right.
    while bounded_end < len(text) - 1 and not text[bounded_end + 1].isspace():
        bounded_end += 1
    
    # Reduce to the left.

    if bounded_end < bounded_start:
        # If the end index is before the start index, swap them.
        bounded_start, bounded_end = bounded_end, bounded_start

    return bounded_start, bounded_end


# def censor_text(text, max_length=128):
#     """
#     Run the text through the model to predict banned tokens.
#     For each token flagged as banned, indicate location so we can send that snippet to the main text model.
#     """
#     # Tokenize the input text.
#     encoding = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding="max_length",
#         max_length=max_length,
#         return_offsets_mapping=True,
#     )
#     offset_mapping = encoding.pop("offset_mapping")[0]  # shape: (seq_length, 2)

#     with torch.no_grad():
#         outputs = text_supp_model(**encoding)
#     # Get predictions: shape (1, seq_length, num_labels).
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)[0].tolist()

#     # Prepare to build the censored text.
#     banned_log = []  # list marking suspicious locations

#     for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping.tolist())):
#         start, end = offset
#         if start == end:
#             continue  # Skip special tokens and padding.
#         if pred == 1:  # If predicted as banned.
#             token_str = text[start:end]
#             banned_log.append(f"{start}-{end}")
#     return banned_log


def saveImgToFile(img, path):
    img.save(path)


@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected image"}), 400

    try:
        img = Image.open(BytesIO(file.read()))
        results = img_model(img)
        predictions = results[0].boxes

        response = {
            "predictions": [
                {
                    "class": img_model.names[int(pred.cls)],
                    "confidence": float(pred.conf),
                }
                for pred in predictions
            ]
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_text", methods=["POST"])
def predict_text():
    if "text" not in request.form:
        return jsonify({"error": "No text part"}), 400

    text = request.form["text"]

    try:
        start_time = time.time()

        inputs = tokenizer(
            text, return_tensors="pt", max_length=MAX_LEN, padding=True, truncation=True
        ).to(device)

        # for word in banned_words:
        #     if word in text.lower():
        #         return {
        #             "class": "profanity",
        #             "confidence": 1.0,
        #         }
            
        for class_name in banned_dict:
            for word in banned_dict[class_name]:
                if word in text.lower():
                    return {
                        "class": class_name,
                        "confidence": 1.0,
                    }

        # for token in tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]):
        #     if token.lower() in banned_words:
        #         okay = False
        #         break
        # if not okay:

        print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

        outputs = text_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        # confidence = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]
        confidence = torch.softmax(outputs.logits, dim=1).tolist()[0]

        class_names = [
            "drugs",
            "explicit",
            "gambling",
            "games",
            # "monetary",
            "profanity",
            "background",
            # "social",
        ]
        idx_to_name = {index: name for index, name in enumerate(class_names)}

        response = {
            "class": idx_to_name[prediction],
            "confidence": confidence[prediction],
        }

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"{response} ({elapsed_time:.4f})")

        # print(response, end_time - start_time)

        return jsonify(response), 200
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/predict_text_supplementary', methods=['POST'])
def predict_text_supplementary():
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
        logits = supp_text_model(batch)  # [batch_size, MAX_LEN, NUM_CLASSES]
        probs = torch.softmax(logits, dim=-1)  # [batch_size, MAX_LEN, NUM_CLASSES]
        max_probs, preds = torch.max(probs, dim=-1)  # both: [batch_size, MAX_LEN]
        # If confidence is low, set the prediction to background (0).
        # print(max_probs)
        preds[max_probs < CONFIDENCE_THRESHOLD] = 0

    token_confidences = max_probs.cpu().tolist()
    preds = preds.cpu().tolist()
    results = []
    for idx, (text, pred_seq) in enumerate(zip(texts, preds)):
        # Extract this sample's token confidences.
        sample_confidences = token_confidences[idx]

        effective_length = min(len(text), MAX_LEN)
        pred_seq = pred_seq[:effective_length]
        sample_confidences = sample_confidences[:effective_length]
        print(f"Decoding: {text}")
        spans = decode_labels(pred_seq, sample_confidences, label_to_category, min_span_length=3)

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
            else:
                text = new_phrase

        results.append({"text": text, "flags": spans})
    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)