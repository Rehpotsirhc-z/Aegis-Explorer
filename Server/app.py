import os

import time
from pathlib import Path

# from transformers import (
#     BertTokenizer,
#     BertForSequenceClassification,
#     BertTokenizerFast,
#     BertForTokenClassification,
# )
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
from PIL import Image
from io import BytesIO

# from torch.quantization import quantize_dynamic
# from dataset import BannedWordDataset
# from text_model import CNNClassifier, MAX_LEN, EMBEDDING_DIM, NUM_CLASSES
import difflib
import json
from openai import OpenAI

try:
    from local_secrets import OPENAI_API_KEY as _OPENAI_API_KEY
except Exception:
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_OPENAI_API_KEY) if _OPENAI_API_KEY else OpenAI()


# Helper: classify texts with OpenAI and return your expected shape
def classify_texts_openai(texts):
    """
    Returns: list of { "text": str, "flags": [ { "category": str, "confidence": float } ] }
    Categories allowed: profanity, explicit, drugs, games, gambling
    """
    # Hard clamp input length for cost/safety
    items = [
        {"id": i, "text": (t if isinstance(t, str) else str(t))[:1000]}
        for i, t in enumerate(texts)
    ]

    system_prompt = (
        "You are a strict K-12 content safety classifier. "
        "For each input item, decide zero or more categories from this exact set: "
        "profanity, explicit, drugs, games, gambling. "
        "Definitions: "
        "- profanity: vulgar or obscene language and slurs; "
        "- explicit: sexual content, sexual acts, nudity, sexting; anything sexual involving minors is explicit; ALSO INCLUDES: graphic violence, gore, severe physical harm, and detailed depictions of injury or death; "
        "- drugs: illegal drugs, misuse of prescription drugs, paraphernalia; "
        "- games: references that primarily direct to or discuss web-based or online games; "
        "- gambling: betting, wagering, casinos, lotteries. "
        "If none apply, return an empty flags array. "
        'Output JSON only with a top-level object: {"results":[{'
        '"id": number, "flags":[{"category": string, "confidence": number}] }...]}. '
        "confidence is a number in [0,1]. Do not rewrite or summarize the text."
    )

    user_payload = {"items": items}

    # Use JSON mode for structured output
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )

    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        # Fallback: everything safe
        return [{"text": t, "flags": []} for t in texts]

    results_by_id = {
        r.get("id"): r for r in data.get("results", []) if isinstance(r, dict)
    }
    allowed = {"profanity", "explicit", "drugs", "games", "gambling"}

    out = []
    for i, t in enumerate(texts):
        entry = results_by_id.get(i, {})
        flags_raw = (
            entry.get("flags", []) if isinstance(entry.get("flags", []), list) else []
        )
        flags = []
        for f in flags_raw:
            try:
                cat = str(f.get("category", "")).lower().strip()
                if cat in allowed:
                    conf = float(f.get("confidence", 0.0))
                    conf = max(0.0, min(1.0, conf))
                    flags.append({"category": cat, "confidence": conf})
            except Exception:
                continue
        out.append({"text": t, "flags": flags})
    return out


# set number of threads to 8
# torch.set_num_threads(8)

WHITELIST_FILE = "whitelist.txt"
BANNED_DIR = "banned"
ALLOWED_WORDS_SET = set()
# Load allowed words from the whitelist file
try:
    with open(WHITELIST_FILE, "r") as f:
        for line in f:
            word = line.strip().lower()
            if word:  # Ensure it's not empty
                ALLOWED_WORDS_SET.add(word)
except FileNotFoundError:
    print(f"Whitelist file missing")
BLACKLIST_FILE = "blacklist.txt"
BLACKLIST_JSON_FILE = "blacklist.json"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# CORPUS_FILE = "corpus.txt"
# MAX_LEN = 256
# CONFIDENCE_THRESHOLD = 0.99  # if max probability is below this, mark as background

# # Load the dataset (to get vocabulary and label mapping)
# dataset_obj = BannedWordDataset(BANNED_DIR, CORPUS_FILE, max_len=MAX_LEN)
# vocab = dataset_obj.vocab
# label_to_category = dataset_obj.label_to_category

# Build model.
# vocab_size = len(vocab)
# supp_text_model = CNNClassifier(vocab_size, EMBEDDING_DIM, NUM_CLASSES, MAX_LEN)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# supp_text_model.load_state_dict(torch.load("cnn_model.pt", map_location=device))
# supp_text_model.to(device)
# supp_text_model.eval()

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# text_model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=6
# )

# text_model.load_state_dict(
#     torch.load("models/text/model_v7.pth", map_location=device),
#     strict=False,
# )

# # text_model = quantize_dynamic(text_model, {torch.nn.Linear}, dtype=torch.qint8)
# text_model.to(device)


# text_model.eval()


def load_word_set(file_path):
    word_set = set()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            word_set = {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Warning: {file_path} not found; ignoring.")
    return word_set


banned_words = load_word_set(BLACKLIST_FILE)
banned_dict = (
    json.load(open(BLACKLIST_JSON_FILE, "r", encoding="utf-8"))
    if Path(BLACKLIST_JSON_FILE).exists()
    else {}
)


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
    spans = [
        span for span in spans if (span["end"] - span["start"] + 1) >= min_span_length
    ]
    return spans


def expand_to_word_boundaries(text, start, end):
    """
    Expands a given span [start, end] to cover the entire word in text.
    A word is defined as a contiguous sequence of non-whitespace characters.
    Returns (new_start, new_end) indices.
    """

    bounded_start = min(start, len(text) - 1)
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


@app.post("/predict_image")
async def predict_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()
        img = Image.open(BytesIO(img_bytes))

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

        return JSONResponse(response)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/predict_text")
async def predict_text(payload: dict):
    if "texts" not in payload:
        return JSONResponse({"error": "Missing 'texts' field"}, status_code=400)

    texts = payload["texts"]
    if not isinstance(texts, list):
        return JSONResponse({"error": "'texts' must be a list"}, status_code=400)

    try:
        results = classify_texts_openai(texts)
        return JSONResponse(results)
    except Exception as e:
        print("OpenAI classification error:", e)
        return JSONResponse([{"text": t, "flags": []} for t in texts])
