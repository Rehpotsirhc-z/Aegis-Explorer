from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Load model weights and move to the selected device
model.load_state_dict(torch.load('model/model.pth', map_location=device), strict=False)
model.to(device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input text from the request
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Tokenize input and move to the correct device
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Perform prediction
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).tolist()[0]
        print(prediction, confidence)

        # Define class labels
        idx_to_name = {
            # 0: "good",
            0: "drugs",
            1: "explicit",
            2: "gambling",
            3: "games",
            # 5: "monetary",
            4: "profanity",
            5: "good",
            # 7: "social",
        }

        # Return the prediction as JSON
        final_prediction = idx_to_name[prediction] if confidence[prediction] > 0.75 else f"good:{idx_to_name[prediction]}"
        return jsonify({'prediction': final_prediction, 'confidence': confidence[prediction]})
    except Exception as e:
        print(f"Error predicting text: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True)
