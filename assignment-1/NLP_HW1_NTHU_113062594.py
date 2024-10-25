# %% [markdown]
# 有用 github copilot 輔助
# ## Part I: Data Pre-processing
# %%
import pandas as pd

# %%
# Download the Google Analogy dataset
# !wget http://download.tensorflow.org/data/questions-words.txt

# %%
# Preprocess the dataset
file_name = "questions-words"
with open(f"{file_name}.txt", "r") as f:
    data = f.read().splitlines()

# %%
# check data from the first 10 entries
for entry in data[:10]:
    print(entry)

# %%
# TODO1: Write your code here for processing data to pd.DataFrame
# Please note that the first five mentions of ": " indicate `semantic`,
# and the remaining nine belong to the `syntatic` category.

questions = []
categories = []
sub_categories = []
current_category = None
current_sub_category = None

for i, line in enumerate(data):
    if line.startswith(": "):
        if "gram" in line:
            current_category = "Syntatic"
        else:
            current_category = "Semantic"
        current_sub_category = line
        continue
    questions.append(line)
    categories.append(current_category)
    sub_categories.append(current_sub_category)


# %%
# Create the dataframe
df = pd.DataFrame(
    {
        "Question": questions,
        "Category": categories,
        "SubCategory": sub_categories,
    }
)

# %%
df.head()

# %%
df.to_csv(f"{file_name}.csv", index=False)

# %% [markdown]
# ## Part II: Use pre-trained word embeddings
# - After finish Part I, you can run Part II code blocks only.

# %%
import gensim.downloader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm

# %%
data = pd.read_csv("questions-words.csv")

# %%
MODEL_NAME = "glove-wiki-gigaword-100"
# You can try other models.
# https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models

# Load the pre-trained model (using GloVe vectors here)
model = gensim.downloader.load(MODEL_NAME)
print("The Gensim model loaded successfully!")

# %%
# Do predictions and preserve the gold answers (word_D)
preds = []
golds = []

for analogy in tqdm(data["Question"]):
    # TODO2: Write your code here to use pre-trained word embeddings for getting predictions of the analogy task.
    # You should also preserve the gold answers during iterations for evaluations later.
    """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
    word_a, word_b, word_c, word_d = analogy.lower().split()
    preds.append(model[word_b] + model[word_c] - model[word_a])
    golds.append(model[word_d])


# %%
# Perform evaluations. You do not need to modify this block!!


def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)


golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

words = set()

# TODO3: Plot t-SNE for the words in the SUB_CATEGORY `: family`
for question, sub_category in zip(data["Question"], data["SubCategory"]):
    if sub_category != SUB_CATEGORY:
        continue

    words.update(question.lower().split())

vectors = []
labels = []

for word in words:
    vectors.append(model[word])
    labels.append(word)

vectors = np.array(vectors)
tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=10)
tsne_vectors = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 6))
plt.title("Word Relationships from Google Analogy Task")
plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1])
for label, x, y in zip(labels, tsne_vectors[:, 0], tsne_vectors[:, 1]):
    plt.annotate(label, xy=(x, y))
# plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")

# %% [markdown]
# ### Part III: Train your own word embeddings

# %% [markdown]
# ### Get the latest English Wikipedia articles and do sampling.
# - Usually, we start from Wikipedia dump (https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). However, the downloading step will take very long. Also, the cleaning step for the Wikipedia corpus ([`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus)) will take much time. Therefore, we provide cleaned files for you.

# %%
# Download the split Wikipedia files
# Each file contain 562365 lines (articles).
# !gdown --id 1jiu9E1NalT2Y8EIuWNa1xf2Tw1f1XuGd -O wiki_texts_part_0.txt.gz
# !gdown --id 1ABblLRd9HXdXvaNv8H9fFq984bhnowoG -O wiki_texts_part_1.txt.gz
# !gdown --id 1z2VFNhpPvCejTP5zyejzKj5YjI_Bn42M -O wiki_texts_part_2.txt.gz
# !gdown --id 1VKjded9BxADRhIoCzXy_W8uzVOTWIf0g -O wiki_texts_part_3.txt.gz
# !gdown --id 16mBeG26m9LzHXdPe8UrijUIc6sHxhknz -O wiki_texts_part_4.txt.gz

# %%
# Download the split Wikipedia files
# Each file contain 562365 lines (articles), except the last file.
# !gdown --id 17JFvxOH-kc-VmvGkhG7p3iSZSpsWdgJI -O wiki_texts_part_5.txt.gz
# !gdown --id 19IvB2vOJRGlrYulnTXlZECR8zT5v550P -O wiki_texts_part_6.txt.gz
# !gdown --id 1sjwO8A2SDOKruv6-8NEq7pEIuQ50ygVV -O wiki_texts_part_7.txt.gz
# !gdown --id 1s7xKWJmyk98Jbq6Fi1scrHy7fr_ellUX -O wiki_texts_part_8.txt.gz
# !gdown --id 17eQXcrvY1cfpKelLbP2BhQKrljnFNykr -O wiki_texts_part_9.txt.gz
# !gdown --id 1J5TAN6bNBiSgTIYiPwzmABvGhAF58h62 -O wiki_texts_part_10.txt.gz

# %%
# Extract the downloaded wiki_texts_parts files.
# !gunzip -k wiki_texts_part_*.gz

# %%
# Combine the extracted wiki_texts_parts files.
# !cat wiki_texts_part_*.txt > wiki_texts_combined.txt

# %%
# Check the first ten lines of the combined file
# !head -n 10 wiki_texts_combined.txt

# %% [markdown]
# Please note that we used the default parameters of [`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus) for cleaning the Wiki raw file. Thus, words with one character were discarded.

# %%
# Now you need to do sampling because the corpus is too big.
# You can further perform analysis with a greater sampling ratio.

import random

wiki_txt_path = "wiki_texts_combined.txt"
# wiki_texts_combined.txt is a text file separated by linebreaks (\n).
# Each row in wiki_texts_combined.txt indicates a Wikipedia article.

with open(wiki_txt_path, "r", encoding="utf-8") as f:
    with open("wiki_texts_combined_0.2.txt", "w", encoding="utf-8") as output_file:
        for line in f:
            if random.random() < 0.2:
                output_file.write(line)
    # TODO4: Sample `20%` Wikipedia articles
    # Write your code here

# %%
# TODO5: Train your own word embeddings with the sampled articles
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
# Hint: You should perform some pre-processing before training.

from functools import partial
from multiprocessing import Pool

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import (
    STOPWORDS,
    preprocess_string,
    remove_stopwords,
    stem_text,
    strip_multiple_whitespaces,
    strip_numeric,
    strip_punctuation,
)


def to_lower(x):
    return x.lower()


with open("wiki_texts_combined_0.2.txt", "r", encoding="utf-8") as f:
    sentences = f.read().splitlines()

stopwords = STOPWORDS - {
    "mostly",
    "he",
    "she",
    "his",
    "her",
    "most",
    "serious",
    "describe",
    "go",
    "move",
    "say",
    "see",
    "computer",
    "find",
}
filters = [
    to_lower,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    partial(remove_stopwords, stopwords=stopwords),
    stem_text,
]

features = []
with Pool(32) as pool:
    sentences = pool.starmap(
        preprocess_string, zip(sentences, [filters] * len(sentences))
    )


print(sentences[:10])


# %%
model = Word2Vec(sentences, min_count=10, workers=32)

# %%
model.save("word2vec_0.2_stem_stop_10.model")

# %%
from gensim.models import Word2Vec

data = pd.read_csv("questions-words.csv")
model = Word2Vec.load("word2vec_0.2_stem_stop_10.model")
vector = model.wv

# %%
# Do predictions and preserve the gold answers (word_D)
from functools import partial

from gensim.parsing.preprocessing import (
    STOPWORDS,
    preprocess_string,
    remove_stopwords,
    stem_text,
    strip_multiple_whitespaces,
    strip_numeric,
    strip_punctuation,
)

preds = []
golds = []

stopwords = STOPWORDS - {
    "mostly",
    "he",
    "she",
    "his",
    "her",
    "most",
    "serious",
    "describe",
    "go",
    "move",
    "say",
    "see",
    "computer",
    "find",
}
filters = [
    lambda x: x.lower(),
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    partial(remove_stopwords, stopwords=stopwords),
    stem_text,
]


for analogy in tqdm(data["Question"]):
    # TODO6: Write your code here to use your trained word embeddings for getting predictions of the analogy task.
    # You should also preserve the gold answers during iterations for evaluations later.
    """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
    word_a, word_b, word_c, word_d = preprocess_string(analogy, filters)
    preds.append(vector[word_b] + vector[word_c] - vector[word_a])
    golds.append(vector[word_d])

# %%
# Perform evaluations. You do not need to modify this block!!


def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)


golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO7: Plot t-SNE for the words in the SUB_CATEGORY `: family`
# Collect words from Google Analogy dataset
words = set()

for question, sub_category in zip(data["Question"], data["SubCategory"]):
    if sub_category != SUB_CATEGORY:
        continue

    question = preprocess_string(question, filters)

    words.update(question)

vectors = []
labels = []

for word in words:
    vectors.append(vector[word])
    labels.append(word)

vectors = np.array(vectors)
tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=10)
tsne_vectors = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 6))
plt.title("Word Relationships from Google Analogy Task")
plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1])
for label, x, y in zip(labels, tsne_vectors[:, 0], tsne_vectors[:, 1]):
    plt.annotate(label, xy=(x, y))
# plt.show()
plt.savefig("word_relationships_self_training.png", bbox_inches="tight")
