from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

model.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')), strict=False)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    # confidence = torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0]


    idx_to_name = {
            0: "good",
            1: "drugs",
            2: "explicit",
            3: "gambling",
            4: "games",
            # 5: "monetary",
            6: "profanity",
            # 7: "social",
    }

    return jsonify({'prediction': idx_to_name[prediction]})

if __name__ == '__main__':
    app.run(debug=True)