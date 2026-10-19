"""
NLTK Showcase — a guided, narration-friendly tour of core NLP steps.
Requirements: pip install nltk
"""

import math
import nltk

from nltk import pos_tag, ne_chunk, FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tree import Tree
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter


# 0. SETUP
REQUIRED_PACKAGES = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
    "wordnet",
    "omw-1.4",
    "vader_lexicon",
]


# Download the required NLTK packages. Downloads are skipped if the data is already available.
def ensure_nltk_data():
    for package in REQUIRED_PACKAGES:
        nltk.download(package, quiet=True)


# HELPER FUNCTION - Print a formatted section heading.
def section(title: str):
    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)


# 1. TOKENIZATION
"""
Break the text into sentences and words.
Returns:
    sentences -> list of sentences
    tokens    -> list of word tokens
"""
def tokenize_text(text: str) -> tuple[list[str], list[str]]:
    section("1. TOKENIZATION — breaking text into sentences and words")
    sentences: list[str] = sent_tokenize(text)             # Sentence tokenization
    tokens: list[str] = word_tokenize(text)                # Word tokenization
    print(f"tokens: {tokens}")
    print(f"Sentences ({len(sentences)}):")

    for sentence in sentences:
        print("  -", sentence)

    print(f"\nFirst 15 word tokens: {tokens[:15]}")
    return sentences, tokens


# 2. STOPWORD REMOVAL
# Convert text to lowercase, tokenize it, remove stopwords, and remove non-alphabetic tokens.
# This function is also used by the cosine similarity phase.
def get_clean_tokens(text: str) -> list[str]:
    tokens: list[str] = word_tokenize(text.lower())
    stop_words: set[str] = set(stopwords.words("english"))

    filtered_tokens: list[str] = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return filtered_tokens


# Remove stopwords from the already tokenized text.
def remove_stopwords(tokens: list[str]) -> list[str]:
    section("2. STOPWORD REMOVAL — filtering out low-information words")
    stop_words: set[str] = set(stopwords.words("english"))

    filtered: list[str] = [
        word
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]

    print(f"Before ({len(tokens)} tokens): {tokens[:12]} ...")
    print(f"After  ({len(filtered)} tokens): {filtered[:12]} ...")
    return filtered


# 3. STEMMING AND LEMMATIZATION - Demonstrate stemming and lemmatization.
def stemming_lemmatization():
    section("3. STEMMING vs LEMMATIZATION — reducing words to their root form")

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    sample_words = ["running", "fantastic", "studies", "wonderfully", "toolkit"]

    print(f"{'Word':<15}{'Stem':<15}{'Lemma':<15}")

    for word in sample_words:
        stem: str = stemmer.stem(word)
        lemma: str = lemmatizer.lemmatize(word)
        print(f"{word:<15}{stem:<15}{lemma:<15}")


# 4. PART-OF-SPEECH TAGGING - Assign a grammatical tag to each word in the first sentence.
def perform_pos_tagging(sentences: list[str]) -> list[tuple[str, str]]:
    section("4. POS TAGGING — labeling each word's grammatical role")
    tagged = pos_tag(word_tokenize(sentences[0]))
    print(tagged)
    return tagged


# 5. NAMED ENTITY RECOGNITION - Identify named entities such as people, organizations, and locations.
def perform_ner(sentences: list[str]) -> Tree:
    section("5. NAMED ENTITY RECOGNITION — finding people, places, organizations")
    # Use the second sentence because it contains "Dr. Sarah Chen" and "Stanford University".
    tagged: list[tuple[str, str]] = pos_tag(word_tokenize(sentences[1]))
    # print(f"tagged: {tagged}")
    entity_tree: Tree = ne_chunk(tagged)
    print(f"entity_tree: {entity_tree}")
    for chunk in entity_tree:
        if hasattr(chunk, "label"):
            entity_text = " ".join(word for word, tag in chunk)
            print(f"  {chunk.label():<10} -> {entity_text}")
    return entity_tree


# 6. FREQUENCY DISTRIBUTION - Find the most frequently occurring words after stopword removal
def frequency_distribution(filtered: list[str]) -> FreqDist:
    section("6. FREQUENCY DISTRIBUTION — which words show up most")
    freq = FreqDist(word.lower() for word in filtered)
    for word, count in freq.most_common(5):
        print(f"  {word:<12} {count}")
    return freq


# 7. SENTIMENT ANALYSIS - Perform sentiment analysis using VADER.
def sentiment_analysis(sentences: list[str]):
    section("7. SENTIMENT ANALYSIS — is the tone positive or negative?")
    sia = SentimentIntensityAnalyzer()
    for sentence in sentences:
        scores: dict[str, float] = sia.polarity_scores(sentence)
        if scores["compound"] > 0.05:
            mood = "positive"
        elif scores["compound"] < -0.05:
            mood = "negative"
        else:
            mood = "neutral"

        print(f"  [{mood:^8}] compound={scores['compound']:+.2f} {sentence}")


# Calculate cosine similarity between two sentences
# 8. COSINE SIMILARITY
def cosine_similarity(doc1: str, doc2: str) -> float:
    # Step 1: Clean and tokenize both documents
    tokens1: list[str] = get_clean_tokens(doc1)
    tokens2: list[str] = get_clean_tokens(doc2)

    # Step 2: Create frequency vectors / Bag-of-Words frequency vectors
    vector1 = Counter(tokens1)
    vector2 = Counter(tokens2)

    # Step 3: Find all unique words
    all_words = set(vector1.keys()).union(set(vector2.keys()))

    # Step 4: Calculate dot product
    dot_product: int = sum(
        vector1[word] * vector2[word]
        for word in all_words)

    # Step 5: Calculate magnitudes
    magnitude1 = math.sqrt(sum(vector1[word] ** 2 for word in all_words))
    magnitude2 = math.sqrt(sum(vector2[word] ** 2 for word in all_words))

    # Step 6: Prevent division by zero
    if not magnitude1 or not magnitude2:
        return 0.0

    # Step 7: Calculate cosine similarity
    similarity: float = dot_product / (magnitude1 * magnitude2)
    return similarity


# Compare every possible pair of sentences
# Sentence 1 vs Sentence 2
# Sentence 1 vs Sentence 3
# Sentence 2 vs Sentence 3
def calculate_sentence_similarity(sentences: list[str]):
    section("8. COSINE SIMILARITY — measuring similarity between sentences")
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity: float = cosine_similarity(sentences[i], sentences[j])

            print(f"\nSentence {i + 1} vs Sentence {j + 1}")
            print(f"Sentence {i + 1}: {sentences[i]}")
            print(f"Sentence {j + 1}: {sentences[j]}")
            print(f"Cosine Similarity: {similarity:.4f}")


if __name__ == "__main__":
    ensure_nltk_data()                  # Setup

    TEXT = (
        "NLTK is a leading platform for building Python programs to work with human language data. "
        "Dr. Sarah Chen from Stanford University said the toolkit makes text analysis wonderfully accessible for beginners. "
        "It is not always the fastest library, but it is a fantastic way to learn how natural language processing actually works under the hood."
    )

    print("INPUT TEXT:\n")
    print(TEXT)


    sentences, tokens = tokenize_text(TEXT)                     # Phase 1: Tokenization
    filtered = remove_stopwords(tokens)                         # Phase 2: Stopword Removal
    stemming_lemmatization()                                    # Phase 3: Stemming and Lemmatization
    perform_pos_tagging(sentences)                              # Phase 4: POS Tagging
    perform_ner(sentences)                                      # Phase 5: Named Entity Recognition
    frequency_distribution(filtered)                            # Phase 6: Frequency Distribution
    sentiment_analysis(sentences)                               # Phase 7: Sentiment Analysis
    calculate_sentence_similarity(sentences)                    # Phase 8: Cosine Similarity
    
    section("DONE — from raw text to structured, analyzable insight.")



#                TEXT
#                  ▼
#           sent_tokenize()
#                  ▼
#              sentences
#       ┌──────────┼──────────┐
#       ▼          ▼          ▼
#   Sentence 1  Sentence 2  Sentence 3
#       │          │          │
#       ▼          ▼          ▼
#  get_clean_tokens()
#       │          │          │
#       ▼          ▼          ▼
#    Counter     Counter     Counter
#       │          │          │
#       └──────────+──────────┘
#                  ▼
#       Compare pairs of sentences
#                  ▼
#       cosine_similarity()




#                     YOUR INPUT
#                         │
#                         ▼
#               ┌───────────────────┐
#               │     TEXT          │
#               │                   │
#               │  Your paragraph   │
#               └─────────┬─────────┘
#                         │
#                         ▼
#           ┌──────────────────────────┐
#           │ tokenize_text(TEXT)      │
#           └────────────┬─────────────┘
#                        │
#              ┌─────────┴─────────┐
#              ▼                   ▼
#        sentences              tokens
#        (list)                 (list)
#              │                   │
#              │                   ▼
#              │        ┌────────────────────┐
#              │        │ remove_stopwords() │
#              │        └─────────┬──────────┘
#              │                  │
#              │                  ▼
#              │           filtered_tokens
#              │
#              ├─────────────────────────────────────┐
#              │                                     │
#              ▼                                     ▼
#    perform_pos_tagging()                    perform_ner()
#              │                                     │
#              ▼                                     ▼
#       (word, POS) pairs                      Entity Tree
#              │                                     │
#              │                              PERSON, ORGANIZATION
#              │                              etc.
#              │
#              ├───────────────┐
#              │               │
#              ▼               ▼
#    frequency_distribution() sentiment_analysis()
#              │               │
#              ▼               ▼
#        Word frequencies   Sentiment scores
#        (FreqDist)         positive/negative/
#                           neutral


# | Tag    | Meaning                               | Example                 |
# | ------ | ------------------------------------- | ----------------------- |
# | `NN`   | Singular noun                         | `book`                  |
# | `NNS`  | Plural noun                           | `books`                 |
# | `NNP`  | Singular proper noun                  | `Sarah`                 |
# | `NNPS` | Plural proper noun                    | `Americans`             |
# | `VB`   | Base verb                             | `run`                   |
# | `VBD`  | Past-tense verb                       | `ran`                   |
# | `VBG`  | Gerund/present participle             | `running`               |
# | `VBN`  | Past participle                       | `eaten`                 |
# | `VBP`  | Non-3rd-person present verb           | `run`                   |
# | `VBZ`  | 3rd-person singular present           | `runs`                  |
# | `JJ`   | Adjective                             | `beautiful`             |
# | `JJR`  | Comparative adjective                 | `better`                |
# | `JJS`  | Superlative adjective                 | `best`                  |
# | `RB`   | Adverb                                | `quickly`               |
# | `RBR`  | Comparative adverb                    | `faster`                |
# | `RBS`  | Superlative adverb                    | `fastest`               |
# | `DT`   | Determiner                            | `the`, `a`              |
# | `IN`   | Preposition/subordinating conjunction | `in`, `from`            |
# | `PRP`  | Personal pronoun                      | `he`, `she`, `it`       |
# | `PRP$` | Possessive pronoun                    | `his`, `her`            |
# | `CC`   | Coordinating conjunction              | `and`, `but`            |
# | `CD`   | Cardinal number                       | `10`, `five`            |
# | `MD`   | Modal                                 | `can`, `should`, `must` |
# | `TO`   | `to`                                  | `to`                    |
# | `UH`   | Interjection                          | `wow`, `oh`             |
# | `WDT`  | Wh-determiner                         | `which`, `that`         |
# | `WP`   | Wh-pronoun                            | `who`, `what`           |
# | `WP$`  | Possessive Wh-pronoun                 | `whose`                 |
# | `WRB`  | Wh-adverb                             | `where`, `when`, `why`  |
# | `EX`   | Existential `there`                   | `There is...`           |
# | `POS`  | Possessive ending                     | `'s`                    |
# | `RP`   | Particle                              | `up` in `pick up`       |
# | `SYM`  | Symbol                                | `$`, `%`                |
# | `FW`   | Foreign word                          | foreign-language term   |