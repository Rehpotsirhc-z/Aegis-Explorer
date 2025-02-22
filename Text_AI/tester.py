from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=6)

# Load model weights and move to the selected device
model.load_state_dict(
    torch.load("model/model_v7.pth", map_location=device), strict=False
)
model.to(device)
model.eval()


# Function to classify text
def classify_text(input_text):
    # Tokenize input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits

    # Convert logits to probabilities
    probabilities = softmax(logits, dim=1).squeeze()

    # Get the predicted class and confidence score
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()
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

    return idx_to_name[predicted_class], predicted_class, confidence


# User input loop
print("Enter text to classify (type 'exit' to quit):")
while True:
    user_input = input("Text: ")
    if user_input.lower() == "exit":
        break

    predicted_class, class_id, confidence = classify_text(user_input)
    print(f"Predicted Class: {predicted_class}, ID: {class_id}, Confidence: {confidence:.2f}")
