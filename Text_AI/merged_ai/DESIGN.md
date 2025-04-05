Merged Behaviors

Dataset:
- source folder has txt's of the different categories
- each txt has 



Model Requirements:
- received sentence as input
- output where it sees a banned word in the format: {"category": "category_name", location: [start, end]}


"Hybrid" approach:
- tokenize and normalize the sentence
- fuzzy match the tokens with the banned words for candidates
- pass the candidates to the model to decide if the word should be flagged
- return the flagged words with the confidence, category, location, and canonical form of the word
- 