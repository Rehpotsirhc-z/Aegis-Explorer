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
import difflib

# set number of threads to 8
# torch.set_num_threads(8)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
image_model_path = Path("models/image/model_v9.pt")
img_model = YOLO(image_model_path)
img_model.to(device)

# Load supplementary model
SUPPLEMENTARY_MODEL_PATH = "models/results"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text_supp_model = BertForTokenClassification.from_pretrained(SUPPLEMENTARY_MODEL_PATH)
text_supp_model.eval()

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
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


def censor_text(text, max_length=128):
    """
    Run the text through the model to predict banned tokens.
    For each token flagged as banned, indicate location so we can send that snippet to the main text model.
    """
    # Tokenize the input text.
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offset_mapping = encoding.pop("offset_mapping")[0]  # shape: (seq_length, 2)

    with torch.no_grad():
        outputs = text_supp_model(**encoding)
    # Get predictions: shape (1, seq_length, num_labels).
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)[0].tolist()

    # Prepare to build the censored text.
    banned_log = []  # list marking suspicious locations

    for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping.tolist())):
        start, end = offset
        if start == end:
            continue  # Skip special tokens and padding.
        if pred == 1:  # If predicted as banned.
            token_str = text[start:end]
            banned_log.append(f"{start}-{end}")
    return banned_log


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
                if float(pred.conf) >= 0.5
            ]
        }

        # save the image with prediction as the name
        file_name = response["predictions"][0]["class"]

        # check if file is aready there, add number to end if so
        # i = 1
        # while os.path.isfile(f"output/{file_name}.jpg"):
        #     file_name = response['predictions'][0]['class'] + str(i)
        #     i+=1
        # print(f"FILENAME: {file_name}")

        # saveImgToFile(img, f"output/{file_name}.jpg")

        # print(img_model.names[int(pred.cls)], float(pred.conf))
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
            text, return_tensors="pt", max_length=128, padding=True, truncation=True
        ).to(device)
        outputs = text_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        # confidence = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]
        confidence = torch.softmax(outputs.logits, dim=1).tolist()[0]

        class_names = {
            "drugs",
            "explicit",
            "gambling",
            "games",
            # "monetary",
            "profanity",
            "background",
            # "social",
        }
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
