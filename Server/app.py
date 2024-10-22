import os
from pathlib import Path
from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, BertTokenizerFast
from flask_cors import CORS
from ultralytics import YOLO
import torch
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load the model
image_model_path = Path("models/image/model_v9.pt")
img_model = YOLO(image_model_path)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
text_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=8
)

text_model.load_state_dict(
    torch.load("models/text/model_v2.pth", map_location=torch.device("cpu")), strict=False
)
text_model.eval()


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

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_text", methods=["POST"])
def predict_text():
    if "text" not in request.form:
        return jsonify({"error": "No text part"}), 400

    text = request.form["text"]

    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = text_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

        confidence = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]

        class_names = {
            "good",
            "drugs",
            "explicit",
            "gambling",
            "games",
            "monetary",
            "profanity",
            "social",
        }
        idx_to_name = {index: name for index, name in enumerate(class_names)}

        response = {
            "class": idx_to_name[prediction],
            "confidence": confidence[prediction],
        }

        print(response)

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
