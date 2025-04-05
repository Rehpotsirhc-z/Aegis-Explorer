import requests

def main():
    url = "http://localhost:5000/predict_batch"
    print("Enter sentences to analyze (type 'exit' to quit):")
    while True:
        # For simplicity, allow user to enter one sentence per request.
        user_input = input("Text: ")
        if user_input.strip().lower() == "exit":
            break
        # Prepare payload as a list (for batch processing, you could add multiple)
        payload = {"texts": [user_input]}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                results = response.json()
                for result in results:
                    print("Original Text:", result["text"])
                    censored_elements = []
                    if result["flags"]:
                        print("Flagged spans:")
                        for span in result["flags"]:
                            start, end, category, conf = span["start"], span["end"], span["category"], span["conf"]
                            flagged = result["text"][start:end+1]
                            print(f"  Characters {start}-{end}: '{flagged}' -> {category} (confidence: {conf})")
                            censored_elements.append((start, end, flagged, category))
                    else:
                        print("No problematic content detected.")
                # print sentence with the flagged sections censored with block chars
                censored_text = user_input
                for start, end, flagged, category in sorted(censored_elements, key=lambda x: x[0], reverse=True):
                    censored_text = censored_text[:start] + "█" * len(flagged) + censored_text[end+1:]
                print("Censored Text:", censored_text)
            else:
                print("Error:", response.status_code, response.text)
        except Exception as e:
            print("An error occurred:", str(e))

if __name__ == "__main__":
    main()
