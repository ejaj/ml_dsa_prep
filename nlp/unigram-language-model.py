from collections import Counter

# Function to tokenize sentences
def tokenize(sentence):
    return sentence.split()

# Function to build the vocabulary and count word frequencies
def build_vocabulary(corpus):
    words = []
    for sentence in corpus:
        words.extend(tokenize(sentence))
    return Counter(words)

# Function to calculate word probabilities
def calculate_word_probabilities(word_counts, total_words):
    return {word: count / total_words for word, count in word_counts.items()}

# Function to calculate the probability of a sentence
def calculate_sentence_probability(sentence, word_probabilities):
    words = tokenize(sentence)
    sentence_probability = 1.0
    for word in words:
        sentence_probability *= word_probabilities.get(word, 0)  # If word is not in vocabulary, P(w) = 0
    return sentence_probability


corpus = [
    "I love machine learning",
    "I love deep learning",
    "Deep learning is amazing",
    "I love machine learning techniques"
]

# Build the vocabulary and word counts
word_counts = build_vocabulary(corpus)

# Calculate word probabilities
total_words = sum(word_counts.values())
word_probabilities = calculate_word_probabilities(word_counts, total_words)

# Define the sentence to calculate the probability for
sentence = "I love learning"

# Calculate sentence probability
sentence_probability = calculate_sentence_probability(sentence, word_probabilities)

# Output the results
print(f"Vocabulary (Word Frequencies): {word_counts}")
print(f"Word Probabilities: {word_probabilities}")
print(f"Probability of the sentence '{sentence}': {sentence_probability}")