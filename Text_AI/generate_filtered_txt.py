from pathlib import Path
word_bank = set()





if __name__ == "__main__":
    file_directory = Path("alt_source")
    file_name = file_directory / "profanity.txt"
    # file_name = "alt_source/profanity.txt"
    word_list = input(f"Enter a comma separated list of keywords to search in {file_name}: ").split(',')
    
    print(file_name)

    for word in word_list:
        word_bank.add(word.strip().lower())

    lines = file_name.read_text().splitlines()

    for line in lines:
        words = line.split()
        for word in words:
            if word.strip().lower() in word_bank:
                (file_directory / "filtered.txt").write_text(line)
                break

    print("Filtered file generated as filtered.txt")