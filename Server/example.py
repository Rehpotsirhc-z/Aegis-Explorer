# server.py
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import difflib

app = Flask(__name__)

# Load the tokenizer and model.
MODEL_PATH = "./results"  # Path to the directory where the fine-tuned model is saved.
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()  # Set the model to evaluation mode.

# --- Load banned words from file ---
def load_banned_words(filepath):
    """Load banned words from a text file, one word per line."""
    with open(filepath, 'r') as f:
        # Store in a set for fast membership testing.
        banned_set = {line.strip().lower() for line in f if line.strip()}
    return banned_set

BANNED_WORDS_FILE = "banned_words.txt"
banned_words_set = load_banned_words(BANNED_WORDS_FILE)
# For fuzzy matching, we'll work with a list.
banned_words_list = list(banned_words_set)

def match_banned_word(token):
    """
    Given a token (as a string), return the banned word from banned_words_list
    that best matches it using fuzzy matching.
    """
    matches = difflib.get_close_matches(token.lower(), banned_words_list, n=1, cutoff=0.6)
    return matches[0] if matches else None

def censor_text(text, max_length=128):
    """
    Run the text through the model to predict banned tokens.
    For each token flagged as banned, replace its characters with block characters
    and log (via fuzzy matching) which banned word it most resembles.
    """
    # Tokenize the input text.
    encoding = tokenizer(text,
                         return_tensors="pt",
                         truncation=True,
                         padding="max_length",
                         max_length=max_length,
                         return_offsets_mapping=True)
    offset_mapping = encoding.pop("offset_mapping")[0]  # shape: (seq_length, 2)

    with torch.no_grad():
        outputs = model(**encoding)
    # Get predictions: shape (1, seq_length, num_labels).
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()

    # Prepare to build the censored text.
    text_chars = list(text)
    banned_log = {}  # Map character span (e.g., "start-end") to the matched banned word.
    
    for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping.tolist())):
        start, end = offset
        if start == end:
            continue  # Skip special tokens and padding.
        if pred == 1:  # If predicted as banned.
            token_str = text[start:end]
            banned_candidate = match_banned_word(token_str)
            banned_log[f"{start}-{end}"] = banned_candidate if banned_candidate is not None else token_str
            # Replace each character in this token span with a block character.
            for i in range(start, end):
                text_chars[i] = "█"
    censored_text = "".join(text_chars)
    return censored_text, banned_log

@app.route("/censor", methods=["POST"])
def censor():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    censored_text, banned_log = censor_text(text)
    response = {
        "censored_text": censored_text,
        "banned_log": banned_log
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)