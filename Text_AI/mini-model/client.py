import requests

def main():
    # url = "http://localhost:5000/predict"
    # command = ""
    # while command.lower() != "exit":
    #     word = input("Enter a word to test: ")
    #     payload = {"word": word}
    #     response = requests.post(url, json=payload)
    #     if response.status_code == 200:
    #         result = response.json()
    #         print(f"Word: {result['word']}")
    #         print(f"Banned: {result['banned']}")
    #         print(f"Confidence: {result['confidence']:.4f}")
    #     else:
    #         print("Error:", response.status_code, response.text)
    url = "http://localhost:5000/predict_batch"
    command = ""
    while command.lower() != "exit":
        # Prompt the user for multiple words, separated by commas.
        words_str = input("Enter words separated by a space: ")
        command = words_str.strip()
        if command.lower() == "exit":
            break
        words = [word.strip() for word in words_str.split() if word.strip()]
        
        payload = {"words": words}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            results = response.json()
            for result in results:
                print(result)
                print(f"Word: {result['word']}")
                print(f"Category: {result['category']}")
                print(f"Confidence: {result['confidence']:.4f}")
                print("-----")
        else:
            print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    main()
