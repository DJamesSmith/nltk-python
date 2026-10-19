# NLTK Showcase: From Raw Text to Real Insight

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.9-F2A541)
![License](https://img.shields.io/badge/license-MIT-green)

A guided, end-to-end tour of classic Natural Language Processing in Python using **NLTK** (the Natural Language Toolkit). One short paragraph goes in, and eight pipeline phases turn it into structured, analyzable output: tokens, filtered keywords, word roots, grammatical tags, named entities, word counts, sentiment scores and sentence similarity.

The program is written to be read and explained. Every phase prints its own labelled section, so you can follow exactly what happens to the text at each step.

> 🎥 **Watch the full walkthrough on YouTube:** [NLTK Tutorial: Natural Language Processing in Python, Step by Step](https://youtube.com/watch?v=YOUR_VIDEO_ID)

---

## Output preview

![Terminal output of nltk_showcase.py showing all eight NLP phases](docs/terminal-output.png)

<p align="center"><sub><i>Terminal output of <code>nltk_showcase.py</code>: each phase prints its own section, from tokenization through to cosine similarity. <a href="docs/terminal-output.png">View full-size screenshot</a></i></sub></p>

---

## The pipeline

| # | Phase | What it does | NLTK tools |
|---|-------|--------------|------------|
| 1 | **Tokenization** | Splits text into sentences and words, handling abbreviations like "Dr." correctly | `sent_tokenize`, `word_tokenize` |
| 2 | **Stopword removal** | Filters out low-information words ("is", "a", "the") and punctuation | `stopwords` corpus |
| 3 | **Stemming vs lemmatization** | Reduces words to a root form: rule-based chopping vs dictionary lookup | `PorterStemmer`, `WordNetLemmatizer` |
| 4 | **POS tagging** | Labels each word's grammatical role using Penn Treebank tags | `pos_tag` |
| 5 | **Named entity recognition** | Finds people, organizations and places | `ne_chunk` |
| 6 | **Frequency distribution** | Counts the most common words after cleaning | `FreqDist` |
| 7 | **Sentiment analysis** | Scores tone as positive, negative or neutral | `SentimentIntensityAnalyzer` (VADER) |
| 8 | **Cosine similarity** | Compares sentences using bag-of-words vectors, implemented from scratch | `collections.Counter`, `math` |

Each phase builds on the ones before it. For example, NER runs on POS-tagged tokens, and cosine similarity reuses the same cleaning step as stopword removal.

---

## Getting started

### Requirements

- Python **3.9 or later** (the code uses built-in generic type hints such as `list[str]`)
- `nltk`

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/nltk-showcase.git
cd nltk-showcase

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install nltk
```

### Run it

```bash
python nltk_showcase.py
```

The first run downloads the required NLTK data packages automatically, so it can take a few extra minutes. Later runs reuse the downloaded data and start almost instantly.

---

## A note on NLTK data

Installing the library and downloading its data are **two separate steps**:

- `pip install nltk` installs the **library** (the code) into your environment.
- `nltk.download(...)` fetches the **data** (trained models, word lists, dictionaries). This data is not a Python package and cannot be installed with pip.

The program handles the second step for you in `ensure_nltk_data()`:

| Resource | Used for |
|----------|----------|
| `punkt`, `punkt_tab` | Sentence and word tokenization |
| `stopwords` | Stopword removal |
| `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng` | POS tagging |
| `maxent_ne_chunker`, `maxent_ne_chunker_tab`, `words` | Named entity recognition |
| `wordnet`, `omw-1.4` | Lemmatization |
| `vader_lexicon` | Sentiment analysis |

Both the classic and `_tab` versions are listed so the program works on older and newer NLTK releases.

By default, NLTK stores this data in a shared `nltk_data` folder in your user directory, **not** inside your virtual environment, so it survives a deleted or recreated venv. To see where it lives on your machine:

```python
import nltk

print(nltk.__file__)                                       # where the library is installed
print(nltk.data.path)                                      # where NLTK looks for data
print(nltk.downloader.Downloader().default_download_dir()) # where downloads go
```

---

## Customizing the input

The sample text lives in the `TEXT` variable inside `if __name__ == "__main__":`. Each of its three sentences has a job:

1. A neutral, factual sentence (POS tagging)
2. A sentence containing a person and an organization (named entity recognition)
3. A sentence with mixed sentiment, "not always the fastest" vs "fantastic" (sentiment analysis)

You can replace it with your own text. Keep at least two sentences, because the NER phase reads the second one (`sentences[1]`).

---

## Limitations worth knowing

This project is about understanding the fundamentals, so the trade-offs are part of the lesson:

- **Bag of words ignores meaning.** "toolkit" and "library" score zero similarity even though they mean nearly the same thing. Word embeddings and transformer models were built to close this gap.
- **No lemmatization before similarity.** "work" and "works" count as different words in Phase 8. Lemmatizing first would improve the score.
- **The lemmatizer defaults to nouns.** `lemmatize("running")` returns `running`; pass `pos="v"` to get `run`.
- **NLTK favors clarity over speed.** For production workloads, consider [spaCy](https://spacy.io/) or transformer-based libraries.

---

## Resources

- [NLTK documentation](https://www.nltk.org/)
- [Natural Language Processing with Python](https://www.nltk.org/book/), the free NLTK book by its creators
- [NLTK HOWTOs and examples](https://www.nltk.org/howto.html)

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.