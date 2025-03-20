import requests

def main():
    url = "http://localhost:5000/predict_batch"
    command = ""
    while command.lower() != "exit":
        # Prompt the user for multiple words or phrases, separated by a space.
        # (If you want to allow phrases with spaces, consider using a delimiter such as a comma.)
        words_str = input("Enter words/phrases separated by a space (or type 'exit' to quit): ")
        command = words_str.strip()
        if command.lower() == "exit":
            break
        # Split by whitespace and strip each token.
        words = [word.strip() for word in words_str.split() if word.strip()]
        
        payload = {"words": words}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            results = response.json()
            for result in results:
                print("-----")
                print(f"Word/Phrase: {result['word']}")
                print(f"Category: {result['category']}")
                print(f"Confidence: {result['confidence']:.4f}")
            print("-----\n")
        else:
            print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    main()
