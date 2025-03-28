from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from dataset import BannedWordDataset, categories

app = Flask(__name__)

# Configuration
BANNED_DIR = "source"           # Directory containing banned term files.
CORPUS_FILE = "corpus.txt"
SCRIPTED_MODEL_PATH = "banned_classifier_torchscript.pt"
MAX_LEN = 128                   # Increased max_len to accommodate longer terms.

# Rebuild the vocabulary using the same settings as training.
dataset = BannedWordDataset(BANNED_DIR, CORPUS_FILE, max_len=MAX_LEN)
vocab = dataset.vocab
vocab_size = len(vocab)

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the TorchScript optimized model
model = torch.jit.load(SCRIPTED_MODEL_PATH, map_location=device)
model.to(device)
model.eval()
print(f"Model loaded on {device}.")

def encode_word(word):
    """Encodes a single word/phrase into a fixed-length tensor and moves it to the correct device."""
    return dataset.encode_word(word).unsqueeze(0).to(device)

def encode_words(words):
    """Encodes a list of words/phrases into a batch tensor and moves it to the correct device."""
    tensors = [dataset.encode_word(word) for word in words]
    batch = torch.stack(tensors).to(device)
    return batch

@app.route('/predict', methods=['POST'])
def predict():
    # Endpoint for single-word prediction.
    data = request.get_json()
    if not data or 'word' not in data:
        return jsonify({"error": "Please provide a 'word' field in the JSON payload."}), 400

    word = data['word'].lower()
    input_tensor = encode_word(word)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted].item()
    
    result = {
        "word": word,
        "category": categories[predicted],
        "confidence": confidence
    }
    return jsonify(result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Endpoint for processing a batch of words/phrases.
    Expects a JSON payload with a "words" field containing a list.
    """
    data = request.get_json()
    if not data or 'words' not in data:
        return jsonify({"error": "Please provide a 'words' field in the JSON payload."}), 400
    
    words = data['words']
    if not isinstance(words, list):
        return jsonify({"error": "'words' should be a list."}), 400
    
    # Lowercase all words for consistency.
    words = [w.lower() for w in words]
    input_tensor = encode_words(words)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        print(probs)
        predicted = torch.argmax(probs, dim=1).tolist()
        confidences = probs.max(dim=1)[0].tolist()
    
    results = []
    for word, pred, conf in zip(words, predicted, confidences):
        results.append({
            "word": word,
            "category": categories[pred],
            "confidence": conf
        })
    return jsonify(results)

if __name__ == '__main__':
    # Run the server on port 5000.
    app.run(host='0.0.0.0', port=5000)
