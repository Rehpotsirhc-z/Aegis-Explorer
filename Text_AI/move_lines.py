def move_lines_containing_keyword(source_file, dest_file, keyword):
    # Read all lines from the source file
    with open(source_file, 'r') as f:
        lines = f.readlines()

    # List to hold the lines that do NOT contain the keyword
    lines_to_keep = []

    # Open the destination file in append mode
    with open(dest_file, 'a') as f_dest:
        # Iterate over each line from the source file
        for line in lines:
            if keyword in line:
                # If the line contains the keyword, write it to the destination file
                f_dest.write(line)
            else:
                # Otherwise, keep it in the list
                lines_to_keep.append(line)

    # Rewrite the source file with only the lines that didn't contain the keyword
    with open(source_file, 'w') as f:
        f.writelines(lines_to_keep)

if __name__ == '__main__':
    # Ask the user for the keyword to search for
    keyword = input("Enter the keyword: ")

    # Specify your file names (adjust these as needed)
    source_file = 'alt_source/profanity_copy.txt'
    dest_file = 'alt_source/explicit_copy.txt'

    # Move the lines containing the keyword from source_file to dest_file
    move_lines_containing_keyword(source_file, dest_file, keyword)
    print("Operation completed.")
