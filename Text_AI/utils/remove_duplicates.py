def remove_duplicate_lines(filename, ignore_case=False):
    """
    Removes duplicate lines from the file.
    
    Parameters:
      filename (str): The path to the file.
      ignore_case (bool): If True, duplicate detection is done in a case-insensitive manner.
    """
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    seen = set()
    unique_lines = []
    for line in lines:
        # Normalize the line by stripping whitespace and newline characters.
        normalized_line = line.strip()
        # If ignoring case, lower-case the normalized version.
        key = normalized_line.lower() if ignore_case else normalized_line
        
        if key not in seen:
            seen.add(key)
            unique_lines.append(line)
    
    with open(filename, 'w', encoding='utf-8') as file:
        file.writelines(unique_lines)

if __name__ == '__main__':
    filename = input("Enter the filename (e.g., file.txt): ")
    # Set ignore_case to True if you want "This is a test" and "this is a test" to be treated as duplicates.
    remove_duplicate_lines(filename, ignore_case=False)
    print("Duplicate lines have been removed.")